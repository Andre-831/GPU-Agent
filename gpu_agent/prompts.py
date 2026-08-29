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
You are an expert GPU performance engineer specializing in PyTorch and Triton.

A generated Triton implementation failed verification against the PyTorch
reference.

## Target GPU
{gpu_specs}

## PyTorch Reference
{pytorch_code}

## Failed Triton Implementation
{triton_code}

## Verification Failure
Type: {error_type}
Error: {error}

## Refinement History
{refinement_history}

## Task
Repair the Triton implementation so that it correctly matches the PyTorch
reference.

Use the current verification failure together with the refinement history to
determine the most likely cause of the failure and make a targeted repair.

When the failure is a correctness error:
- Analyze the numerical diagnostics before modifying the kernel.
- Re-derive the PyTorch operation's mathematical semantics from the reference.
- Check indexing, tensor layout, strides, padding, grouping, parameter layout,
  reduction order, and accumulation precision where applicable.
- Determine whether the mismatch is caused by incorrect computation or
  floating-point precision.
- If the output is numerically close but outside tolerance, investigate
  accumulation precision and operation ordering rather than making unrelated
  structural changes.
- Compare the current failure with previous correctness failures in the
  refinement history.
- Preserve changes that reduced numerical error or otherwise improved
  verification.
- Do not revert to an earlier approach that produced a worse verification
  result unless there is a specific technical reason to do so.
- Do not make performance optimizations until correctness is restored.

When the failure is a compilation or runtime error:
- Use the exact compiler/runtime error to identify the invalid operation or
  resource constraint.
- Fix the reported failure before making unrelated changes.
- Preserve previous changes that successfully resolved earlier failures.
- Consider Triton constraints such as valid block sizes, power-of-two
  requirements, shared-memory usage, register pressure, launch configuration,
  and supported APIs when relevant.

Use the refinement history as evidence:
- Do not repeat a previous attempted fix that failed to resolve the problem.
- Do not reintroduce an error that was already fixed in an earlier round.
- Prefer changes that continue a demonstrated improvement from previous rounds.
- If a previous change improved correctness but was insufficient, build on that
  change rather than discarding it.
- If several rounds show no improvement, reconsider the kernel's underlying
  algorithm, indexing, or mapping to the PyTorch operation instead of making
  small variations of the same unsuccessful approach.

Requirements:
- Fix the correctness, compilation, or runtime problem.
- Preserve the exact computation performed by the PyTorch reference.
- Use only valid Triton APIs.
- Preserve a class named `ModelNew` that subclasses `torch.nn.Module`.
- ModelNew must have the same constructor interface as the reference Model.
- ModelNew.forward() must have the same input/output interface as the reference
  Model.forward().
- If the reference Model contains parameters or buffers, ModelNew must create
  corresponding parameters or buffers so deterministic initialization with the
  same random seed produces equivalent model state.
- ModelNew.forward() must execute the Triton implementation.
- Do not use torch operations to perform the core computation being replaced.
- Torch may be used for tensor allocation, shape/stride inspection, parameters,
  buffers, and model/interface bookkeeping.
- Do not hardcode outputs or exploit known input values.
- Return Python code only.
- Use minimal comments.
"""