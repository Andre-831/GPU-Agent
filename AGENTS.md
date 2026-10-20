# GPU-Agent — Agent Instructions

## Project Overview
GPU-Agent converts PyTorch workloads into correctness-verified Triton kernels and iteratively optimizes them using GPU benchmarks, NVIDIA profiling tools, Roofline heuristics, and LLM feedback. The current evaluation target is KernelBench Level 1; arbitrary workloads and fusion are future goals. Read README.md before architectural changes.

## Repository Map
- main.py: CLI entry point for a KernelBench workload.
- evaluate_problems.py: sequential Level 1 problem evaluation.
- gpu_agent/prompts.py: shared generation, repair, and optimization prompts.
- gpu_agent/generation/: LLM generation, repair, and correctness verification.
- gpu_agent/optimization/: orchestration, benchmarking, profiling, Roofline analysis, and optimization.
- gpu_agent/analysis/: diagnostics and performance analysis.
- gpu_agent/evaluation/: final kernel evaluation.
- tests/: automated regression tests.

## Existing Pipeline
1. Generate four independent Triton seeds, with LLM work overlapping.
2. Serialize GPU correctness verification; repair failed seeds within bounded attempts.
3. Benchmark verified seeds sequentially and select the lowest-median-latency seed.
4. Profile the current best kernel with Nsys and NCU and analyze Roofline metrics.
5. Generate an optimized candidate, verify it, and benchmark only if correct.
6. Keep the best verified candidate across up to five optimization rounds.
7. Save generated_kernel_best.py and run final multi-trial evaluation.

Preserve these invariants unless the requested change explicitly modifies them.

## Correctness and Performance
- PyTorch is the correctness reference. Never accept or benchmark an unverified candidate.
- Preserve the best verified kernel if a challenger is incorrect or slower.
- Do not bypass verification or weaken numerical tolerances to make an optimization pass.
- GPU verification and benchmarking must not contend on a single GPU.
- Base performance claims on measured results, not model predictions.
- Current selection compares median latencies; paired incumbent/challenger remeasurement and statistical confidence are NOT implemented. Do not claim otherwise.
- Keep hardware-specific conclusions and heuristic Roofline classifications appropriately qualified.
- Preserve error handling, subprocess isolation, and timeouts.

## LLM and Generated Code
- Reuse the existing OpenAI client and prompts; do not hardcode secrets or silently change model identifiers.
- Treat model output as untrusted executable code until verified.
- Preserve relevant GPU specifications, numerical precision guidance, errors, and refinement history.
- Avoid unnecessary API requests. Unit tests must not make live OpenAI calls.
- Generated files such as generated_kernel.py, generated_kernel_best.py, generated_kernel_seed_*.py, and generated_kernel_v*.py are runtime artifacts, not primary source.
- Optimization runs use shared artifact filenames; do not run multiple jobs concurrently or overwrite results without considering existing artifacts.

## Working Agreement
1. Read relevant files and nested AGENTS.md instructions before editing.
2. For significant changes, explain the intended approach and affected pipeline stages.
3. Prefer focused changes and reuse existing utilities. Avoid unrelated refactors.
4. Add or update tests for success, failure, timeout, and best-kernel preservation paths as relevant.
5. Run targeted tests and, when feasible, python -m pytest tests/ -q.
6. Distinguish CPU/mocked tests from real CUDA/Nsys/NCU validation.
7. Report changes, validation, and limitations accurately.
8. Do not commit, push, or delete important files without explicit authorization.

## Environment
Full end-to-end runs require an NVIDIA CUDA GPU, PyTorch, Triton, KernelBench, Nsys, NCU, and OPENAI_API_KEY. The README documents example commands and the Colab setup.

## Planned, Not Implemented
Broader KernelBench coverage, robust paired latency comparison, richer search, arbitrary PyTorch support, and graph/subgraph fusion are future directions. Do not implement them without a task requesting them.
