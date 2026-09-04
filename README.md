# GPU-Agent

An autonomous GPU kernel generation and optimization system that converts PyTorch workloads into verified Triton kernels, profiles their performance, diagnoses GPU bottlenecks, and iteratively improves generated implementations.

Currently being developed and evaluated using [KernelBench](https://github.com/ScalingIntelligence/KernelBench), a standardized benchmark suite for evaluating LLM-generated GPU kernels against PyTorch workloads.

## How It Works

```text
PyTorch Workload
      ↓
Generate 4 Independent Triton Seeds in Parallel
      ↓
Verify Each Seed (Serialized on the GPU)
      ↓
Repair Failed Seeds and Re-verify
      ↓
Benchmark Verified Seeds Sequentially
      ↓
Select Fastest Seed
      ↓
Nsys + NCU Profiling
      ↓
Roofline Analysis
      ↓
LLM Bottleneck Diagnosis
      ↓
Generate Optimized Triton Kernel
      ↓
Verify + Benchmark
      ↓
Keep Best Kernel
      ↺
```

GPU-Agent combines **LLM-based kernel generation** with real GPU performance data. Four independent seed lifecycles run concurrently so LLM generation and repair requests can overlap. GPU verification is serialized to avoid contention on a single GPU, and every generated kernel must pass correctness verification before it can be benchmarked or accepted.

The fastest verified seed becomes the starting point for profiling and iterative optimization.

During optimization, [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) collect hardware performance data. Roofline analysis uses these metrics to classify the kernel as **compute-bound, memory-bound, or underutilized** and estimate optimization headroom.

The profiling results, hardware metrics, Roofline classification, and kernel implementation are provided to the LLM, which diagnoses likely bottlenecks and generates a new Triton implementation. Each optimization must pass correctness verification and outperform the current best implementation before it can replace it.

## Current Progress

GPU-Agent currently supports:

- **PyTorch → Triton** kernel generation
- Four parallel kernel-generation trajectories
- Automatic correctness verification
- LLM-based repair of incorrect kernels
- Serialized GPU verification
- Isolated seed failures and deterministic candidate ordering
- Verification subprocess timeouts
- Candidate benchmarking and fastest-seed selection
- **NVIDIA Nsight Systems (Nsys)** profiling
- **NVIDIA Nsight Compute (NCU)** hardware metrics
- Roofline bottleneck analysis
- LLM-based performance diagnosis
- Iterative profile → optimize → verify → benchmark loops
- Best-so-far kernel selection
- Multi-trial final correctness evaluation
- **KernelBench Level 1** evaluation

GPU-Agent has been tested end-to-end on KernelBench Level 1 workloads, including parallel candidate generation, correctness verification, GPU profiling, Roofline analysis, iterative optimization, and final evaluation.

## Usage

### Requirements

GPU-Agent requires:

- CUDA-capable NVIDIA GPU
- Python
- PyTorch
- Triton
- OpenAI API key
- NVIDIA Nsight Systems (`nsys`)
- NVIDIA Nsight Compute (`ncu`)
- KernelBench

Confirm that both NVIDIA profiling tools are available:

```bash
nsys --version
ncu --version
```

Set the OpenAI API key in your environment:

```bash
export OPENAI_API_KEY='your-api-key'
```

Run GPU-Agent on a KernelBench problem:

```bash
python main.py external/KernelBench/KernelBench/level1/19_ReLU.py
```

The agent will generate multiple Triton implementations, verify and benchmark them, select the fastest valid seed, profile it, and iteratively attempt to improve its performance.

The best correctness-verified kernel found during the run is saved to:

```text
generated_kernel_best.py
```

## Optimization Loop

For each optimization round, GPU-Agent:

1. Profiles the current best Triton kernel.
2. Collects Nsys and NCU performance metrics.
3. Performs Roofline analysis.
4. Sends the kernel and performance diagnostics to the LLM.
5. Generates an optimized Triton implementation.
6. Verifies the candidate for correctness.
7. Benchmarks the candidate against the current best.
8. Accepts the candidate only if it is correct and faster.

This allows the system to use measured GPU behavior rather than relying solely on the LLM's assumptions about kernel performance.

## Evaluation

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) is used as the current standardized evaluation suite.

KernelBench provides PyTorch workloads that can be used to evaluate:

- Kernel correctness
- Triton generation success rate
- Repair success rate
- Performance relative to PyTorch
- Optimization effectiveness

Current development is focused on **KernelBench Level 1** workloads before expanding to more complex fused operations and model-level workloads.

## What's Next

## What's Next

- Robust benchmarking and kernel selection
- Full KernelBench Level 1 evaluation
- Better optimization search and refinement
- Arbitrary PyTorch workload support
- PyTorch graph/subgraph extraction and fusion
- Evaluation on more complex KernelBench workloads

## Goal

The long-term goal is to build an **autonomous GPU performance agent** that can take arbitrary PyTorch code, identify optimization opportunities, generate optimized Triton kernels, verify their correctness, profile them on real hardware, diagnose performance bottlenecks, and automatically improve their performance.
