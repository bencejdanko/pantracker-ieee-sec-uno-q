// infer_bench <model.pte> [-n runs] [-w warmup] [-t threads] [-e etdump.bin] [-i input.bin ...]
// Loads a .pte through libinfer, feeds fixed inputs (raw files, or a deterministic pattern),
// prints latency statistics (CLOCK_MONOTONIC) as one JSON line.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

extern "C" {
int infer_load(const char*, int, int, void**);
int infer_run(void*, const void**, void**);
int infer_info(void*, char*, int);
int infer_dump(void*, const char*);
void infer_free(void*);
}

static double now_ms() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static std::vector<size_t> parse_nbytes(const std::string& json, bool inputs) {
  std::vector<size_t> r;
  size_t o = json.find("\"outputs\"");
  std::string sec = inputs ? json.substr(0, o) : json.substr(o);
  for (size_t q = 0; (q = sec.find("\"nbytes\":", q)) != std::string::npos; q += 9)
    r.push_back(strtoull(sec.c_str() + q + 9, nullptr, 10));
  return r;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s model.pte [-n N] [-w W] [-t T] [-e etdump] [-i in.bin]...\n", argv[0]); return 2; }
  const char* pte = argv[1];
  int n = 200, w = 20, t = 0;
  const char* etd = nullptr;
  std::vector<const char*> infiles;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-n") n = atoi(argv[++i]);
    else if (a == "-w") w = atoi(argv[++i]);
    else if (a == "-t") t = atoi(argv[++i]);
    else if (a == "-e") etd = argv[++i];
    else if (a == "-i") infiles.push_back(argv[++i]);
  }
  double t0 = now_ms();
  void* h;
  int rc = infer_load(pte, t, etd != nullptr, &h);
  if (rc) { fprintf(stderr, "load failed rc=%d\n", rc); return 1; }
  double load_ms = now_ms() - t0;
  char info[8192];
  infer_info(h, info, sizeof info);
  std::string js(info);
  auto in_sz = parse_nbytes(js, true);
  auto out_sz = parse_nbytes(js, false);
  std::vector<std::vector<unsigned char>> ib(in_sz.size()), ob(out_sz.size());
  for (size_t i = 0; i < in_sz.size(); ++i) {
    ib[i].resize(in_sz[i]);
    if (i < infiles.size()) {
      FILE* f = fopen(infiles[i], "rb");
      if (!f || fread(ib[i].data(), 1, in_sz[i], f) != in_sz[i]) { fprintf(stderr, "bad input %s\n", infiles[i]); return 1; }
      fclose(f);
    } else {  // deterministic smooth pattern (float32 assumed when size%4==0)
      if (in_sz[i] % 4 == 0) {
        float* p = (float*)ib[i].data();
        for (size_t k = 0; k < in_sz[i] / 4; ++k) p[k] = 0.5f * sinf(0.001f * k);
      } else for (size_t k = 0; k < in_sz[i]; ++k) ib[i][k] = (unsigned char)(k * 7);
    }
  }
  for (size_t i = 0; i < out_sz.size(); ++i) ob[i].resize(out_sz[i]);
  std::vector<const void*> ip; for (auto& b : ib) ip.push_back(b.data());
  std::vector<void*> op; for (auto& b : ob) op.push_back(b.data());
  for (int i = 0; i < w; ++i) if ((rc = infer_run(h, ip.data(), op.data()))) { fprintf(stderr, "run failed rc=%d\n", rc); return 1; }
  std::vector<double> lat;
  double s0 = now_ms();
  for (int i = 0; i < n; ++i) {
    double a = now_ms();
    if ((rc = infer_run(h, ip.data(), op.data()))) { fprintf(stderr, "run failed rc=%d\n", rc); return 1; }
    lat.push_back(now_ms() - a);
  }
  double total = now_ms() - s0;
  if (etd) infer_dump(h, etd);
  if (getenv("INFER_BENCH_OUT")) {  // dump outputs for parity checks
    FILE* f = fopen(getenv("INFER_BENCH_OUT"), "wb");
    for (auto& b : ob) fwrite(b.data(), 1, b.size(), f);
    fclose(f);
  }
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) { return lat.empty() ? 0 : lat[std::min(lat.size() - 1, (size_t)(p * lat.size()))]; };
  double mean = 0; for (double x : lat) mean += x; mean = lat.empty() ? 0 : mean / lat.size();
  printf("{\"model\":\"%s\",\"runs\":%d,\"threads\":%d,\"load_ms\":%.2f,\"p50_ms\":%.3f,\"p95_ms\":%.3f,"
         "\"mean_ms\":%.3f,\"min_ms\":%.3f,\"fps\":%.2f,\"io\":%s}\n",
         pte, n, t, load_ms, pct(0.5), pct(0.95), mean, lat.empty() ? 0 : lat[0], n ? n * 1000.0 / total : 0, info);
  infer_free(h);
  return 0;
}
