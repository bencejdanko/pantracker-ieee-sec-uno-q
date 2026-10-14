cmake_minimum_required(VERSION 3.24)
project(infer C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ET_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../.cache/executorch" CACHE PATH "ExecuTorch v1.5.1 source")

set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS ON CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_EVENT_TRACER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "" FORCE)
add_subdirectory(${ET_SRC} executorch)

set(ET_LIBS
  executorch extension_module_static extension_data_loader extension_tensor
  extension_threadpool optimized_native_cpu_ops_lib quantized_ops_lib
  xnnpack_backend vulkan_backend etdump flatccrt)

add_library(infer SHARED infer.cpp)
target_link_libraries(infer PRIVATE ${ET_LIBS})
target_include_directories(infer PRIVATE ${ET_SRC}/..)
set_target_properties(infer PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

add_executable(infer_bench infer_bench.cpp)
target_link_libraries(infer_bench PRIVATE infer)
set_target_properties(infer_bench PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
