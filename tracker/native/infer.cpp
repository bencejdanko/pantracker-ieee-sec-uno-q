// libinfer.so — ExecuTorch 1.5.1 runtime (XNNPACK + Vulkan + portable/optimized kernels + ETDump)
// behind a tiny C ABI, for ctypes (tracker/python/infer.py) and infer_bench.
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

namespace {

struct Spec {
  ScalarType dtype;
  std::vector<executorch::aten::SizesType> sizes;
  size_t nbytes;
};

struct Handle {
  std::unique_ptr<Module> module;
  executorch::etdump::ETDumpGen* etdump = nullptr;  // owned by module
  std::vector<Spec> in, out;
};

const char* dtype_name(ScalarType t) {
  switch (t) {
    case ScalarType::Float: return "float32";
    case ScalarType::Half: return "float16";
    case ScalarType::Char: return "int8";
    case ScalarType::Byte: return "uint8";
    case ScalarType::Int: return "int32";
    case ScalarType::Long: return "int64";
    case ScalarType::Bool: return "bool";
    default: return "unknown";
  }
}

size_t elem_size(ScalarType t) {
  switch (t) {
    case ScalarType::Float: case ScalarType::Int: return 4;
    case ScalarType::Half: return 2;
    case ScalarType::Long: return 8;
    default: return 1;
  }
}

}  // namespace

extern "C" {

// C++ exceptions (the Vulkan backend throws vkapi::Error) must not cross the C ABI into ctypes.
#define INFER_TRY try {
#define INFER_CATCH(name)                                                  \
  }                                                                        \
  catch (const std::exception& ex) {                                       \
    fprintf(stderr, "%s: exception: %s\n", name, ex.what());               \
    return -100;                                                           \
  }                                                                        \
  catch (...) {                                                            \
    fprintf(stderr, "%s: unknown exception\n", name);                      \
    return -101;                                                           \
  }

static int infer_load_impl(const char* pte, int n_threads, int etdump, void** h_out) {
  if (n_threads > 0)
    executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(n_threads);
  auto h = std::make_unique<Handle>();
  std::unique_ptr<executorch::runtime::EventTracer> tracer;
  if (etdump) {
    auto g = std::make_unique<executorch::etdump::ETDumpGen>();
    h->etdump = g.get();
    tracer = std::move(g);
  }
  h->module = std::make_unique<Module>(pte, Module::LoadMode::Mmap, std::move(tracer));
  Error e = h->module->load_method("forward");
  if (e != Error::Ok) {
    fprintf(stderr, "infer_load: load_method(forward) failed for %s: 0x%x\n", pte, (int)e);
    return (int)e ? (int)e : -1;
  }
  auto meta = h->module->method_meta("forward");
  if (!meta.ok()) return (int)meta.error();
  for (int pass = 0; pass < 2; ++pass) {
    size_t n = pass == 0 ? meta->num_inputs() : meta->num_outputs();
    for (size_t i = 0; i < n; ++i) {
      auto ti = pass == 0 ? meta->input_tensor_meta(i) : meta->output_tensor_meta(i);
      if (!ti.ok()) return (int)ti.error();
      Spec s{ti->scalar_type(), {}, ti->nbytes()};
      for (auto d : ti->sizes()) s.sizes.push_back(d);
      (pass == 0 ? h->in : h->out).push_back(std::move(s));
    }
  }
  *h_out = h.release();
  return 0;
}

int infer_load(const char* pte, int n_threads, int etdump, void** h_out) {
  INFER_TRY return infer_load_impl(pte, n_threads, etdump, h_out);
  INFER_CATCH("infer_load")
}

static int infer_run_impl(void* hv, const void** in, void** out);

// in[i] / out[i]: caller-owned contiguous buffers with the dtype/shape from infer_info.
int infer_run(void* hv, const void** in, void** out) {
  INFER_TRY return infer_run_impl(hv, in, out);
  INFER_CATCH("infer_run")
}

static int infer_run_impl(void* hv, const void** in, void** out) {
  auto* h = static_cast<Handle*>(hv);
  std::vector<TensorPtr> keep;
  std::vector<EValue> args;
  for (size_t i = 0; i < h->in.size(); ++i) {
    auto& s = h->in[i];
    keep.push_back(executorch::extension::from_blob(const_cast<void*>(in[i]), s.sizes, s.dtype));
    args.emplace_back(*keep.back());
  }
  auto r = h->module->execute("forward", args);
  if (!r.ok()) {
    fprintf(stderr, "infer_run: execute failed: 0x%x\n", (int)r.error());
    return (int)r.error();
  }
  for (size_t i = 0; i < h->out.size() && i < r->size(); ++i) {
    if (!(*r)[i].isTensor()) continue;
    const auto& t = (*r)[i].toTensor();
    size_t n = t.nbytes() < h->out[i].nbytes ? t.nbytes() : h->out[i].nbytes;
    std::memcpy(out[i], t.const_data_ptr(), n);
  }
  return 0;
}

int infer_info(void* hv, char* json, int cap) {
  auto* h = static_cast<Handle*>(hv);
  std::string s = "{";
  for (int pass = 0; pass < 2; ++pass) {
    auto& v = pass == 0 ? h->in : h->out;
    s += pass == 0 ? "\"inputs\":[" : ",\"outputs\":[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) s += ",";
      s += "{\"dtype\":\"";
      s += dtype_name(v[i].dtype);
      s += "\",\"shape\":[";
      for (size_t j = 0; j < v[i].sizes.size(); ++j) {
        if (j) s += ",";
        s += std::to_string(v[i].sizes[j]);
      }
      s += "],\"nbytes\":" + std::to_string(v[i].nbytes) + "}";
    }
    s += "]";
  }
  s += "}";
  if ((int)s.size() + 1 > cap) return -(int)(s.size() + 1);
  std::memcpy(json, s.c_str(), s.size() + 1);
  return 0;
}

int infer_dump(void* hv, const char* path) {
  auto* h = static_cast<Handle*>(hv);
  if (!h->etdump) return -1;
  auto res = h->etdump->get_etdump_data();
  if (!res.buf || !res.size) return -2;
  FILE* f = fopen(path, "wb");
  if (!f) { free(res.buf); return -3; }
  fwrite(res.buf, 1, res.size, f);
  fclose(f);
  free(res.buf);
  return 0;
}

void infer_free(void* hv) { delete static_cast<Handle*>(hv); }

}  // extern "C"
