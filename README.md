# GPU-Agent

An autonomous GPU kernel generation and optimization system that converts PyTorch workloads into verified Triton kernels, profiles their performance, diagnoses GPU bottlenecks, and iteratively improves generated implementations.

Currently being developed and evaluated using [KernelBench](https://github.com/ScalingIntelligence/KernelBench), a standardized benchmark suite for evaluating LLM-generated GPU kernels against PyTorch workloads.

The project is inspired by [Meta/PyTorch KernelAgent](https://github.com/meta-pytorch/KernelAgent). KernelBench is the current evaluation interface; the long-term goal is to support arbitrary PyTorch workloads.

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
Nsys + NCU Profiling of Current Best Kernel
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
- Adaptive `triton.testing.do_bench` timing with raw samples and statistics
- Benchmark progress reporting and subprocess timeouts
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

Adjust the problem path to your KernelBench checkout. For the current Colab layout:

```bash
cd /content/GPU-Agent
python main.py /content/KernelBench/KernelBench/level1/19_ReLU.py
```

Replace the problem filename to test another workload. Run problems sequentially; generated kernel and profiling files use shared filenames and are overwritten by later runs. Copy any results you want to keep before starting another problem.

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
5. Generates, saves, and prints an optimized Triton implementation.
6. Verifies the candidate for correctness, allowing one correction attempt if verification fails.
7. Benchmarks the verified candidate and compares its median latency with the stored current-best measurement.
8. Accepts the candidate only if verification passed and its median latency is lower; otherwise retains the current best.

This allows the system to use measured GPU behavior rather than relying solely on the LLM's assumptions about kernel performance.

The winning seed is labeled **V1**. Up to five optimization rounds generate **V2–V6**. After optimization, the best kernel is saved and then undergoes final multi-trial correctness and performance evaluation. A final evaluation failure is reported; saving the file does not imply that this final check passed.

### Reading the logs

`OPTIMIZATION V4` marks the start of round V4, before V4 has been generated. If V3 was rejected and V2 remains the best kernel, the profiling output beneath that heading describes **V2**. The agent uses those measurements to generate V4, prints its code under `GENERATED TRITON V4`, verifies it, and only then benchmarks it under `V4 BENCHMARK`.

### Benchmarking and selection

Each initial benchmark times PyTorch first, then the Triton candidate, using approximately **25 ms warmup** and **100 ms measurement** per implementation. `do_bench` adapts the sample count to runtime, so slower workloads can return only a few samples. These budgets exclude some setup and calibration overhead and are not wall-clock limits.

Results include raw timings, median, mean, standard deviation, minimum, maximum, and sample count. Selection uses the Triton median; reported speedup is the PyTorch median divided by the Triton median. KernelBench benchmark workers have a default **300-second subprocess timeout** and print live progress.

**Current limitation:** both seed selection and optimization accept any strictly lower median. The collected spread statistics do not yet affect selection, and the incumbent is not freshly remeasured against each challenger. Tiny differences can therefore reflect measurement noise. Robust paired comparison remains planned, not implemented.

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

- Test the current pipeline on more KernelBench Level 1 problems
- Robust incumbent/challenger comparison with fresh paired measurements, alternating order, and bounded remeasurement
- Full KernelBench Level 1 evaluation
- Better optimization search and refinement
- Arbitrary PyTorch workload support
- PyTorch graph/subgraph extraction and fusion
- Evaluation on more complex KernelBench workloads

## Goal

The long-term goal is to build an **autonomous GPU performance agent** that can take arbitrary PyTorch code, identify optimization opportunities, generate optimized Triton kernels, verify their correctness, profile them on real hardware, diagnose performance bottlenecks, and automatically improve their performance.
