import torch


def es_fp8_blockwise_scaled_grouped_mm(
    output,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_d,
    problem_sizes,
    expert_offsets,
    workspace,
):
    torch.ops.sgl_kernel.es_fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_d,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def es_sm100_mxfp8_blockscaled_grouped_mm(
    output, a, b, sfa, sfb, problem_sizes, expert_offsets, blockscale_offsets
):
    # This sgl-kernel was built with --force only H100 (SM90) support.
    # SM100/MXFP8 expert-specialization kernels were excluded at compile time
    # and the torch.ops registration was removed. This cannot be changed by
    # environment, CMAKE_ARGS, or any build flag.
    raise RuntimeError(
        "es_sm100_mxfp8_blockscaled_grouped_mm is not available. "
        "This build of sgl-kernel is HARD FORCED to support ONLY H100 (compute capability 90 / Hopper). "
        "Blackwell (SM100+) expert specialization features (MXFP8) have been permanently removed from the build. "
        "There is no way to enable them via external settings."
    )


def es_sm100_mxfp8_blockscaled_grouped_quant(
    input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor
):
    raise RuntimeError(
        "es_sm100_mxfp8_blockscaled_grouped_quant is not available. "
        "This build of sgl-kernel is HARD FORCED to support ONLY H100 (compute capability 90 / Hopper). "
        "Blackwell (SM100+) expert specialization features (MXFP8) have been permanently removed from the build. "
        "There is no way to enable them via external settings."
    )
