
An autonomous GPU kernel generation and optimization system that converts PyTorch workloads into verified Triton kernels, profiles their performance, diagnoses GPU bottlenecks, and iteratively improves generated implementations.

Currently being developed and evaluated using **KernelBench** as a standardized suite of PyTorch GPU workloads.

## How It Works

```text
PyTorch Workload
      ↓
Generate Triton Kernel
      ↓
Verify Correctness
      ↓
Repair if Incorrect
      ↓
Benchmark
      ↓
Nsys + NCU Profiling
      ↓
Roofline Analysis
      ↓
LLM-Guided Optimization
      ↓
Verify + Benchmark
      ↓
Keep Best Kernel
```

GPU-Agent combines **LLM-based kernel generation** with real GPU profiling data. Generated kernels must pass correctness verification before being optimized or accepted.

During optimization, **Nsight Systems** and **Nsight Compute** collect hardware performance data. Roofline analysis identifies whether the kernel is **compute-bound, memory-bound, or underutilized**, and this information is provided to the LLM to guide the next optimization.

## Current Progress

GPU-Agent currently supports:

* **PyTorch → Triton** kernel generation
* Automatic correctness verification
* LLM-based repair of incorrect kernels
* **Nsys and NCU** profiling
* Roofline bottleneck analysis
* Iterative LLM-guided optimization
* Best-kernel selection and final evaluation
* **KernelBench Level 1** evaluation

The agent has successfully generated and verified multiple KernelBench workloads.

## Usage

Run GPU-Agent on a KernelBench problem:

```bash
python main.py external/KernelBench/KernelBench/level1/19_ReLU.py
```

The best verified kernel found during the run is saved to:

```text
generated_kernel_best.py
```

## What's Next

* Multi-worker / multi-seed kernel generation
* Verification and compilation timeouts
* Full KernelBench Level 1 evaluation
* Better optimization search and refinement
* Arbitrary PyTorch workload support
* PyTorch graph/subgraph extraction
* Triton kernel fusion

## Goal

The long-term goal is to build an **autonomous GPU performance agent** that can take PyTorch code, generate optimized GPU kernels, verify their correctness, profile them on real hardware, diagnose performance bottlenecks, and automatically improve them.
