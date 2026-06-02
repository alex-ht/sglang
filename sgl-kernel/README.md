# sglang-kernel (prior sgl-kernel)

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) for LLM inference engines

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sglang-kernel)](https://pypi.org/project/sglang-kernel)

</div>

`sglang-kernel` provides optimized compute primitives for LLM inference engines, enabling efficient inference for large language models and vision-language models through custom kernel operations. The source tree remains under the `sgl-kernel/` directory and the Python import path remains `sgl_kernel`.

## Installation
Requires torch == 2.9.1

```bash
# Latest version
pip3 install sglang-kernel --upgrade
```

## Building from Source
Requires
- CMake ≥3.31,
- Python ≥3.10
- scikit-build-core
- ninja(optional)

### Use Makefile to build from the sgl-kernel source tree

```bash
make build
```

### Limit build resource usage (CPU / parallelism)

By default, `make build` (and wheel builds) uses the most memory-efficient settings:
- 1 parallel job (`MAX_JOBS=1`, `CMAKE_BUILD_PARALLEL_LEVEL=1`)
- `SGL_KERNEL_COMPILE_THREADS=1` (NVCC --threads)
- Optimization level `-O1` (via `SGL_KERNEL_OPT_LEVEL=1`; nvcc only accepts numeric levels)
- Only H100/Hopper (SM90 + SM90a) gencodes; no pre-Hopper or Blackwell unless you opt in.

This makes default compilation the least memory hungry.

To build faster when you have sufficient host memory:

```bash
# Use more parallelism
make build MAX_JOBS=8

# Also raise NVCC threads (each nvcc can use more cores internally)
make build MAX_JOBS=8 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=4"

# To also enable support for older GPUs (sm80/sm89 etc):
make build CMAKE_ARGS="-DENABLE_BELOW_SM90=ON"

# To enable Blackwell (SM100+) support (will build extra variant + gencodes):
make build CMAKE_ARGS="-DSGL_KERNEL_ENABLE_SM100A=ON -DSGL_KERNEL_BUILD_SM100_VARIANT=ON"

# Use -Os (optimize for size) on host C++ code (nvcc still gets a numeric level like -O2).
# Note: CUDA device kernels use numeric level only; add -Xcompiler=-Os if you also want it for nvcc's host part.
make build CMAKE_ARGS="-DSGL_KERNEL_OPT_LEVEL=s"

# Or combine multiple:
make build MAX_JOBS=4 CMAKE_ARGS="-DSGL_KERNEL_OPT_LEVEL=s -DSGL_KERNEL_COMPILE_THREADS=2"
```

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from sgl_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## FAQ
- Q: Segmentation fault with CUDA 12.6
- A: Update ptxas to 12.8, reference: [segment fault error](https://github.com/Dao-AILab/flash-attention/issues/1453)
