BOTTLENECK_PROMPT = """\
You are a GPU performance expert analyzing GPU kernel profiling data.

## Task
Analyze the NCU metrics and identify the primary performance bottleneck.

Classify it as:
- memory: Memory bandwidth is the limiting factor
- compute: Compute throughput is the limiting factor
- underutilized: Neither is saturated (<60% both), indicating possible stalls,
  occupancy limitations, instruction dependencies, launch configuration issues,
  or insufficient parallelism.

## GPU Specifications
{gpu_specs}

## Roofline Analysis
- Bottleneck: {roofline_bottleneck}
- Compute SOL: {compute_sol:.1f}%
- Memory SOL: {memory_sol:.1f}%
- Efficiency: {efficiency:.1f}%
- Headroom: {headroom:.1f}%
- At Roofline: {at_roofline}

## NCU Metrics
{ncu_metrics}

## Output
Return JSON only.

Requirements:
- Ground every conclusion in the provided metrics.
- NEVER invent metric names or values.
- Do not assume low occupancy alone indicates poor performance.
- Do not recommend increasing occupancy unless evidence shows occupancy is limiting performance.
- Consider whether the kernel is already near its hardware limit.
- Prioritize fixes most likely to improve measured performance.
"""


TRITON_GENERATION_PROMPT = """\
You are an expert GPU performance engineer specializing in PyTorch and Triton.

Your task is to replace the provided PyTorch computation with an optimized
Triton implementation compatible with the KernelBench model interface.

## Target GPU

{gpu_specs}

## PyTorch Reference

{pytorch_code}

## Requirements

- Preserve the exact computation performed by the PyTorch reference.
- Generate a Triton implementation optimized for the target GPU.
- Include all required imports.
- Include the required @triton.jit kernel or kernels.
- Define a class named `ModelNew` that subclasses `torch.nn.Module`.
- `ModelNew` must have the same constructor interface as the reference `Model`.
- `ModelNew.forward()` must have the same input/output interface as the reference `Model.forward()`.
- If the reference Model contains parameters or buffers, ModelNew must create
  corresponding parameters or buffers so that deterministic initialization with
  the same random seed produces equivalent model state.
- ModelNew.forward() must execute the replacement Triton implementation.
- You may define helper Python functions such as `triton_implementation` if useful,
  but ModelNew is the required entry point.
- Do not use torch operations to perform the core computation being replaced.
- Torch may be used for tensor allocation, shape/stride inspection, parameters,
  buffers, and other model/interface bookkeeping.
- Do not hardcode outputs or exploit known input values.
- Return Python code only.
- Use minimal comments.
- Do not include explanations, tests, or benchmarks.
"""


TRITON_OPTIMIZATION_PROMPT = """\
You are an expert GPU performance engineer specializing in Triton.

## PyTorch Reference
{pytorch_code}

## Current Triton Implementation
{triton_code}

## Target GPU
{gpu_specs}

## Benchmark
PyTorch: {pytorch_ms:.4f} ms
Triton: {triton_ms:.4f} ms
Speedup: {speedup:.2f}x

## Roofline
Classification: {classification}
Compute SOL: {compute_sol:.2f}%
Memory SOL: {memory_sol:.2f}%
Efficiency: {efficiency:.2f}%
Headroom: {headroom:.2f}%

## NCU Metrics
{ncu_metrics}

## Previous Failure
{error}

## Task
Generate an improved, valid Triton implementation compatible with the
KernelBench model interface.

If a previous compiler/runtime error is provided, fix that error before
attempting further optimization.

Requirements:
- Preserve the exact computation performed by the PyTorch reference.
- Preserve correctness.
- Use only valid Triton APIs.
- Preserve a class named `ModelNew` that subclasses `torch.nn.Module`.
- ModelNew must preserve the same constructor interface as the reference Model.
- ModelNew.forward() must preserve the same input/output interface as the
  reference Model.forward().
- If the reference Model contains parameters or buffers, ModelNew must create
  corresponding parameters or buffers so that deterministic initialization with
  the same random seed produces equivalent model state.
- ModelNew.forward() must execute the optimized Triton implementation.
- You may define helper Python functions such as `triton_implementation` if useful.
- Do not use torch operations to perform the core computation being replaced.
- Torch may be used for tensor allocation, shape/stride inspection, parameters,
  buffers, and other model/interface bookkeeping.
- Do not hardcode outputs or exploit known input values.
- Return Python code only.
- Use minimal comments.
"""

TRITON_REPAIR_PROMPT = """\
{triton_guidelines}

## Task

Repair the following Triton implementation based on the verification results.

## PyTorch / KernelBench Problem

{pytorch_code}

## Current Triton Implementation

{triton_code}

## Verification Results

Error type: {error_type}
Error: {error}

## Previous Attempts

{refinement_history}

## Requirements

- Analyze the verification failure and make a targeted repair.
- Preserve the exact computation performed by the PyTorch reference.
- Learn from previous attempts and do not repeat approaches that clearly regressed or failed.
- Follow the Triton programming guidelines above.
- Use only valid Triton APIs.
- Preserve a class named `ModelNew` that subclasses `torch.nn.Module`.
- `ModelNew` must have the same constructor interface as the reference `Model`.
- `ModelNew.forward()` must have the same input/output interface as the reference `Model.forward()`.
- If the reference contains parameters or buffers, preserve equivalent deterministic initialization.
- `ModelNew.forward()` must execute the Triton implementation.
- Keep the core numerical computation inside Triton kernels.
- Focus on correctness before performance optimization.
- Do not hardcode outputs or exploit known input values.
- Return complete Python code only.
"""

TRITON_GUIDELINES = """\
## Triton Programming Guidelines

- Use @triton.jit for Triton kernels.
- Use tl.constexpr for compile-time constants where appropriate.
- Use tl.load() and tl.store() with correct masking.
- Coalesce memory accesses when possible.
- Use tl.program_id() and tl.arange() for indexing.
- Use tl.cdiv() for ceiling division.
- Handle boundary conditions with masks.
- Prefer power-of-two block sizes where required by Triton.
- Use tl.dot() for matrix-style reductions when appropriate.
- Accumulate reductions in FP32 when needed for numerical accuracy.
- Avoid excessive compile-time loop unrolling that can cause very large
  generated kernels or excessive compilation time.
- Choose block sizes, num_warps, and num_stages appropriate for the workload.
- Keep core numerical computation inside Triton kernels.
- PyTorch may only be used for allocation, parameters/buffers, shape/stride
  inspection, dtype/device checks, and launch/interface bookkeeping.
"""