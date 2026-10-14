#!/usr/bin/env bash
# Cross-build libinfer.so + infer_bench for gatos (aarch64 Debian 13). Output: tracker/native/out/
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../.." && pwd)
[ -d "$repo/.cache/executorch" ] || {
  git clone --depth 1 --branch v1.5.1 https://github.com/pytorch/executorch.git "$repo/.cache/executorch"
  git -C "$repo/.cache/executorch" submodule update --init --depth 1 --jobs 8 \
    backends/vulkan/third-party/{Vulkan-Headers,VulkanMemoryAllocator,volk} \
    backends/xnnpack/third-party/{FP16,FXdiv,XNNPACK,cpuinfo,pthreadpool} \
    kernels/optimized/third-party/eigen third-party/{flatbuffers,flatcc,json,pocketfft,gflags,prelude,ao} shim
}
# GCC 14 fixes for v1.5.1 sources (missing <algorithm> includes); idempotent.
for f in $(grep -rlE "std::(find|rotate|sort|max_element|min_element|remove|reverse|count|any_of|all_of)\\b" "$repo/.cache/executorch/backends/vulkan/runtime" --include=*.cpp | sed "s|$repo/.cache/executorch/||"); do
  grep -q '<algorithm>' "$repo/.cache/executorch/$f" || sed -i '0,/^#include/s//#include <algorithm>\n#include/' "$repo/.cache/executorch/$f"
done
# Turnip exposes a Vulkan 1.0 *device*; the instance is created at 1.1 but VK_KHR_get_physical_device_properties2
# is not enabled, so the *KHR entry points resolve to NULL and every 16-bit feature reads as false (=> fp16 models
# silently run fp32 activations). Fall back to the core 1.1 entry points. Idempotent.
dev="$repo/.cache/executorch/backends/vulkan/runtime/vk_api/Device.cpp"
grep -q 'core-1.1 fallback' "$dev" || python3 - "$dev" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
for fn in ("Features2", "Properties2"):
    old = f'instance, "vkGetPhysicalDevice{fn}KHR");'
    new = (f'instance, "vkGetPhysicalDevice{fn}KHR");\n  if (vkGetPhysicalDevice{fn}KHR == nullptr)  // core-1.1 fallback\n'
           f'    vkGetPhysicalDevice{fn}KHR = (PFN_vkGetPhysicalDevice{fn}KHR)vkGetInstanceProcAddr(\n'
           f'        instance, "vkGetPhysicalDevice{fn}");')
    assert old in s, old
    s = s.replace(old, new)
open(p, "w").write(s)
PY
docker build -t infer-xbuild "$here"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$repo:/src" -w /src/tracker/native infer-xbuild bash -c '
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/src/tracker/native/aarch64-linux-gnu.cmake -DPYTHON_EXECUTABLE=/opt/py/bin/python3 &&
  cmake --build build -j"$(nproc)" --target infer infer_bench -- -k 0 &&
  mkdir -p out && cp build/libinfer.so build/infer_bench out/ && aarch64-linux-gnu-strip out/libinfer.so out/infer_bench &&
  ls -l out && file out/*'
