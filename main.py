import subprocess
import sys
import csv
import io
import json
import os
from google import genai
import torch
import triton
import importlib.util
from openai import OpenAI

client = OpenAI()

def profile(workload):
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--output=gpuagent_profile",
        "--force-overwrite=true",
    ] + workload

    print("Profiling: " + " ".join(workload))

    subprocess.run(command, check=True)

    print("\nProfile saved to gpuagent_profile.nsys-rep")


def get_stats(report):
    command = [
        "nsys",
        "stats",
        "--force-export=true",
        "--report",
        report,
        "gpuagent_profile.nsys-rep",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout
NCU_METRICS = [
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__shared_mem_per_block_allocated",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]
def profile_kernel_with_ncu(workload, kernel_name):

    command = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--kernel-name", kernel_name,
        "--launch-count", "1",
        "--metrics", ",".join(NCU_METRICS),
    ] + workload

    print(f"\nDeep profiling kernel: {kernel_name}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def parse_ncu_output(output):
    lines = output.splitlines()

    # Find NCU's actual CSV header.
    # Raw-page output begins with columns such as:
    # ID, Process ID, Process Name, ..., Kernel Name, ...
    header_index = None

    for i, line in enumerate(lines):
        if line.startswith('"ID",') or line.startswith("ID,"):
            header_index = i
            break

    if header_index is None:
        return {}

    csv_text = "\n".join(lines[header_index:])
    reader = csv.DictReader(io.StringIO(csv_text))

    rows = list(reader)

    if not rows:
        return {}

    # We currently profile one kernel launch, so use its row.
    row = rows[-1]

    metrics = {}

    for name in NCU_METRICS:
        value = row.get(name)

        if value is None or value == "":
            continue

        value = value.replace(",", "").replace("%", "")

        try:
            metrics[name] = float(value)
        except ValueError:
            continue

    return metrics



def analyze_roofline(metrics):
    compute_sol = metrics.get(
      "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )

    memory_sol = metrics.get(
      "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    )

    if compute_sol is None or memory_sol is None:
      return {
        "classification": "unknown",
        "compute_sol": compute_sol,
        "memory_sol": memory_sol,
        "efficiency": None,
        "at_roofline": False,
        "headroom": None,
      }

    if compute_sol < 60 and memory_sol < 60:
        classification = "underutilized"
    elif memory_sol >= compute_sol:
        classification = "memory_bound"
    else:
        classification = "compute_bound"

    efficiency = max(compute_sol, memory_sol)

    return {
        "classification": classification,
        "compute_sol": compute_sol,
        "memory_sol": memory_sol,
        "efficiency": efficiency,
        "at_roofline": efficiency >= 95,
        "headroom": max(0, 100 - efficiency),
    }


def parse_kernel_stats(stats):
    kernels = []

    for line in stats.splitlines():
        parts = line.split()

        if len(parts) < 9:
            continue

        try:
            time_percent = float(parts[0])
            total_time_ns = int(parts[1].replace(",", ""))
            instances = int(parts[2])
            avg_time_ns = float(parts[3].replace(",", ""))
            median_time_ns = float(parts[4].replace(",", ""))
            min_time_ns = int(parts[5].replace(",", ""))
            max_time_ns = int(parts[6].replace(",", ""))
            stddev_time_ns = float(parts[7].replace(",", ""))

        except ValueError:
            continue

        name = " ".join(parts[8:])

        kernels.append({
            "time_percent": time_percent,
            "total_time_ns": total_time_ns,
            "instances": instances,
            "avg_time_ns": avg_time_ns,
            "median_time_ns": median_time_ns,
            "min_time_ns": min_time_ns,
            "max_time_ns": max_time_ns,
            "stddev_time_ns": stddev_time_ns,
            "name": name,
        })

    return kernels


def parse_cuda_api_stats(stats):
    cuda_calls = []

    for line in stats.splitlines():
        parts = line.split()

        if len(parts) < 9:
            continue

        try:
            time_percent = float(parts[0])
            total_time_ns = int(parts[1].replace(",", ""))
            num_calls = int(parts[2])
            avg_time_ns = float(parts[3].replace(",", ""))
            median_time_ns = float(parts[4].replace(",", ""))
            min_time_ns = int(parts[5].replace(",", ""))
            max_time_ns = int(parts[6].replace(",", ""))
            stddev_time_ns = float(parts[7].replace(",", ""))

        except ValueError:
            continue

        name = " ".join(parts[8:])

        cuda_calls.append({
            "time_percent": time_percent,
            "total_time_ns": total_time_ns,
            "num_calls": num_calls,
            "avg_time_ns": avg_time_ns,
            "median_time_ns": median_time_ns,
            "min_time_ns": min_time_ns,
            "max_time_ns": max_time_ns,
            "stddev_time_ns": stddev_time_ns,
            "name": name,
        })

    return cuda_calls



def parse_memory_stats(stats):
    memory_ops = []

    for line in stats.splitlines():
        parts = line.split()

        if len(parts) < 9:
            continue

        try:
            time_percent = float(parts[0])
            total_time_ns = int(parts[1].replace(",", ""))
            count = int(parts[2])
            avg_time_ns = float(parts[3].replace(",", ""))
            median_time_ns = float(parts[4].replace(",", ""))
            min_time_ns = int(parts[5].replace(",", ""))
            max_time_ns = int(parts[6].replace(",", ""))
            stddev_time_ns = float(parts[7].replace(",", ""))

        except ValueError:
            continue

        operation = " ".join(parts[8:])

        memory_ops.append({
            "time_percent": time_percent,
            "total_time_ns": total_time_ns,
            "count": count,
            "avg_time_ns": avg_time_ns,
            "median_time_ns": median_time_ns,
            "min_time_ns": min_time_ns,
            "max_time_ns": max_time_ns,
            "stddev_time_ns": stddev_time_ns,
            "operation": operation,
        })

    return memory_ops


def parse_nvtx_stats(stats):
    nvtx_ranges = []

    for line in stats.splitlines():
        parts = line.split()

        # NVTX has 8 numeric columns + Style + Range
        if len(parts) < 10:
            continue

        try:
            time_percent = float(parts[0])
            total_time_ns = int(parts[1].replace(",", ""))
            instances = int(parts[2])
            avg_time_ns = float(parts[3].replace(",", ""))
            median_time_ns = float(parts[4].replace(",", ""))
            min_time_ns = int(parts[5].replace(",", ""))
            max_time_ns = int(parts[6].replace(",", ""))
            stddev_time_ns = float(parts[7].replace(",", ""))

        except ValueError:
            continue

        style = parts[8]

        name = " ".join(parts[9:]).lstrip(":")

        nvtx_ranges.append({
            "time_percent": time_percent,
            "total_time_ns": total_time_ns,
            "instances": instances,
            "avg_time_ns": avg_time_ns,
            "median_time_ns": median_time_ns,
            "min_time_ns": min_time_ns,
            "max_time_ns": max_time_ns,
            "stddev_time_ns": stddev_time_ns,
            "style": style,
            "name": name,
        })

    return nvtx_ranges


def parse_nvtx_kernel_stats(stats):
    nvtx_kernels = []

    for line in stats.splitlines():
        parts = line.split()

        if len(parts) < 13:
            continue

        try:
            nvtx_range = parts[0].lstrip(":")
            style = parts[1]
            pid = int(parts[2].replace(",", ""))
            tid = int(parts[3].replace(",", ""))
            nvtx_instances = int(parts[4])
            kernel_instances = int(parts[5])
            total_time_ns = int(parts[6].replace(",", ""))
            avg_time_ns = float(parts[7].replace(",", ""))
            median_time_ns = float(parts[8].replace(",", ""))
            min_time_ns = int(parts[9].replace(",", ""))
            max_time_ns = int(parts[10].replace(",", ""))
            stddev_time_ns = float(parts[11].replace(",", ""))

        except ValueError:
            continue

        kernel_name = " ".join(parts[12:])

        nvtx_kernels.append({
            "nvtx_range": nvtx_range,
            "style": style,
            "pid": pid,
            "tid": tid,
            "nvtx_instances": nvtx_instances,
            "kernel_instances": kernel_instances,
            "total_time_ns": total_time_ns,
            "avg_time_ns": avg_time_ns,
            "median_time_ns": median_time_ns,
            "min_time_ns": min_time_ns,
            "max_time_ns": max_time_ns,
            "stddev_time_ns": stddev_time_ns,
            "kernel_name": kernel_name,
        })

    return nvtx_kernels


def analyze_profile(profile_data):
    print("\n================ GPU PERFORMANCE ANALYSIS ================")

    kernels = profile_data["kernels"]
    cuda_calls = profile_data["cuda_calls"]
    memory_ops = profile_data["memory_ops"]
    nvtx_ranges = profile_data["nvtx_ranges"]
    nvtx_kernels = profile_data["nvtx_kernels"]

    # --------------------------------------------------
    # Top GPU kernels
    # --------------------------------------------------

    print("\nTop GPU Kernels:")

    for kernel in kernels[:5]:
        total_ms = kernel["total_time_ns"] / 1_000_000

        print(
            f"  {kernel['name']}"
            f" | {total_ms:.2f} ms"
            f" | {kernel['instances']} launches"
            f" | {kernel['time_percent']:.1f}% of kernel time"
        )

    # --------------------------------------------------
    # Memory transfers
    # --------------------------------------------------

    print("\nMemory Operations:")

    for operation in memory_ops:
        total_ms = operation["total_time_ns"] / 1_000_000

        print(
            f"  {operation['operation']}"
            f" | {operation['count']} operations"
            f" | {total_ms:.2f} ms"
        )

    # --------------------------------------------------
    # CUDA API calls
    # --------------------------------------------------

    print("\nTop CUDA API Calls:")

    for call in cuda_calls[:5]:
        total_ms = call["total_time_ns"] / 1_000_000

        print(
            f"  {call['name']}"
            f" | {call['num_calls']} calls"
            f" | {total_ms:.2f} ms"
        )

    # --------------------------------------------------
    # Application regions
    # --------------------------------------------------

    print("\nApplication Regions:")

    for region in nvtx_ranges:
        total_ms = region["total_time_ns"] / 1_000_000

        print(
            f"  {region['name']}"
            f" | {total_ms:.2f} ms"
        )

    # --------------------------------------------------
    # Kernels inside application regions
    # --------------------------------------------------

    print("\nKernels by Application Region:")

    for kernel in nvtx_kernels:
        total_ms = kernel["total_time_ns"] / 1_000_000

        print(
            f"  {kernel['nvtx_range']}"
            f" -> {kernel['kernel_name']}"
            f" | {kernel['kernel_instances']} launches"
            f" | {total_ms:.2f} ms GPU time"
        )


def diagnose_profile(profile_data):
    findings = []

    kernels = profile_data["kernels"]
    cuda_calls = profile_data["cuda_calls"]
    memory_ops = profile_data["memory_ops"]

    # Check synchronization
    for call in cuda_calls:
        if call["name"] == "cudaDeviceSynchronize":
            findings.append({
                "category": "synchronization",
                "severity": "info",
                "evidence": (
                    f"{call['num_calls']} cudaDeviceSynchronize calls "
                    f"took {call['total_time_ns'] / 1_000_000:.2f} ms"
                ),
                "actionable": False,
                "reason": (
                    "Synchronization time may represent the CPU waiting "
                    "for useful GPU work, so high time alone does not "
                    "prove a performance problem."
                ),
            })

    # Check memory transfers
    for operation in memory_ops:
        findings.append({
            "category": "memory_transfer",
            "severity": "investigate",
            "evidence": (
                f"{operation['operation']} occurred "
                f"{operation['count']} times and took "
                f"{operation['total_time_ns'] / 1_000_000:.2f} ms"
            ),
            "actionable": None,
            "reason": (
                "Memory transfers can be expensive, but more context "
                "is needed to determine whether they are necessary."
            ),
        })

    # Check dominant kernels
    for kernel in kernels:
        if kernel["time_percent"] >= 50: #flags kernels usiing >=50% of gpu kernel time
            findings.append({
                "category": "dominant_kernel",
                "severity": "investigate",
                "evidence": (
                    f"{kernel['name']} accounts for "
                    f"{kernel['time_percent']:.1f}% of GPU kernel time"
                ),
                "actionable": None,
                "reason": (
                    "A dominant kernel is worth investigating, but high "
                    "runtime does not mean the kernel is inefficient."
                ),
            })

    return findings


def diagnose_kernel(kernel, ncu_metrics):
    """
    Combine NSYS kernel-level importance with NCU hardware metrics.

    This is intentionally a small V1 heuristic layer. Later, these
    structured facts will be passed to the AI reasoning layer instead
    of growing into a large set of hard-coded rules.
    """
    diagnosis = {
        "kernel": kernel["name"],
        "kernel_time_percent": kernel["time_percent"],
        "priority": "investigate",
        "evidence": [],
        "reason": "",
    }

    compute = ncu_metrics.get("compute_throughput")
    memory = ncu_metrics.get("memory_throughput")
    dram = ncu_metrics.get("dram_throughput")
    achieved_occupancy = ncu_metrics.get("achieved_occupancy")
    theoretical_occupancy = ncu_metrics.get("theoretical_occupancy")
    registers = ncu_metrics.get("registers_per_thread")

    diagnosis["evidence"].append(
        f"{kernel['name']} accounts for {kernel['time_percent']:.1f}% of GPU kernel time"
    )

    if compute is not None:
        diagnosis["evidence"].append(
            f"Compute (SM) throughput is {compute:.2f}%"
        )

    if memory is not None:
        diagnosis["evidence"].append(
            f"Memory throughput is {memory:.2f}%"
        )

    if dram is not None:
        diagnosis["evidence"].append(
            f"DRAM throughput is {dram:.2f}%"
        )

    if achieved_occupancy is not None:
        diagnosis["evidence"].append(
            f"Achieved occupancy is {achieved_occupancy:.2f}%"
        )

    if theoretical_occupancy is not None:
        diagnosis["evidence"].append(
            f"Theoretical occupancy is {theoretical_occupancy:.2f}%"
        )

    if registers is not None:
        diagnosis["evidence"].append(
            f"Registers per thread is {registers:.0f}"
        )

    # V1 judgment:
    # If a dominant kernel is already driving compute very hard,
    # low occupancy alone is not enough evidence that rewriting it
    # should be the first optimization target.
    if compute is not None and compute >= 90:
        diagnosis["priority"] = "low"
        diagnosis["reason"] = (
            "The kernel dominates runtime, but it is already using most of the "
            "GPU's available compute throughput. Low occupancy by itself does "
            "not prove the kernel is inefficient, so rewriting this kernel "
            "should not be the first optimization target."
        )
    else:
        diagnosis["reason"] = (
            "The kernel is important to overall runtime and NCU does not show "
            "near-saturated compute throughput, so it remains worth deeper investigation."
        )

    return diagnosis




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


def analyze_with_llm(kernel_name, ncu_metrics, roofline):
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    prompt = BOTTLENECK_PROMPT.format(
        gpu_specs=json.dumps(gpu_specs, indent=2),
        roofline_bottleneck=roofline["classification"],
        compute_sol=roofline["compute_sol"],
        memory_sol=roofline["memory_sol"],
        efficiency=roofline["efficiency"],
        headroom=roofline["headroom"],
        at_roofline=roofline["at_roofline"],
        ncu_metrics=json.dumps(ncu_metrics, indent=2),
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text



def get_gpu_specs():
  props = torch.cuda.get_device_properties(0)

  return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
        "max_threads_per_block": props.max_threads_per_block,
        "warp_size": props.warp_size,
    }



TRITON_GENERATION_PROMPT = """\
You are an expert GPU performance engineer specializing in PyTorch and Triton.

Your task is to replace the provided PyTorch computation with an optimized
Triton implementation.

## Target GPU

{gpu_specs}

## PyTorch Reference

{pytorch_code}

## Requirements

- Preserve the exact computation performed by the PyTorch reference.
- Generate a Triton implementation optimized for the target GPU.
- Include all required imports.
- Include the @triton.jit kernel.
- Include a Python wrapper named `triton_implementation`.
- `triton_implementation` must preserve the same input/output interface as the PyTorch reference.
- Do not use torch operations to perform the computation inside the replacement.
- Return Python code only.
- Use minimal comments.
- Do not include explanations, tests, or benchmarks.
"""


def generate_triton_kernel(pytorch_code, gpu_specs):

    prompt = TRITON_GENERATION_PROMPT.format(
        pytorch_code=pytorch_code,
        gpu_specs=json.dumps(gpu_specs, indent=2),
    )

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return response.output_text




def extract_python_code(response):
    response = response.strip()

    if response.startswith("```python"):
        response = response[len("```python"):]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()



def verify_candidate(filename="generated_kernel.py"):
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            "candidate_kernel",
            filename
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_shapes = [
            (1024,),
            (4096,),
            (256, 256),
            (12345,),
        ]

        for shape in test_shapes:
            x = torch.randn(shape, device="cuda")

            expected = torch.relu(x)
            actual = module.triton_implementation(x)

            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )

            except AssertionError as e:
                return {
                    "passed": False,
                    "error_type": "correctness",
                    "shape": shape,
                    "error": str(e),
                }

        return {
            "passed": True,
            "tests": len(test_shapes),
        }

    except Exception as e:
        return {
            "passed": False,
            "error_type": "execution",
            "error": str(e),
        }



def benchmark_candidate(filename="generated_kernel.py"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "candidate_kernel",
        filename
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    x = torch.randn(10_000_000, device="cuda")

    # Warmup
    for _ in range(10):
        torch.relu(x)
        module.triton_implementation(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # PyTorch
    start.record()

    for _ in range(100):
        torch.relu(x)

    end.record()
    torch.cuda.synchronize()

    pytorch_ms = start.elapsed_time(end) / 100

    # Triton
    start.record()

    for _ in range(100):
        module.triton_implementation(x)

    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / 100

    return {
        "pytorch_ms": pytorch_ms,
        "triton_ms": triton_ms,
        "speedup": pytorch_ms / triton_ms,
    }



def create_candidate_workload():
    code = """
import torch
from generated_kernel import triton_implementation

x = torch.randn(10_000_000, device="cuda")

# Compile / warmup
for _ in range(5):
    y = triton_implementation(x)

torch.cuda.synchronize()

# Profiling run
y = triton_implementation(x)
torch.cuda.synchronize()
"""

    with open("candidate_workload.py", "w") as f:
        f.write(code)


def profile_candidate():
    workload = ["python", "candidate_workload.py"]

    # Run generated Triton workload under NSYS
    profile(workload)

    # Get and parse kernel stats
    kernel_stats = get_stats("cuda_gpu_kern_sum")
    kernels = parse_kernel_stats(kernel_stats)

    if not kernels:
        print("No candidate GPU kernels found.")
        return None

    print("\n================ CANDIDATE KERNELS ================")

    for kernel in kernels:
      total_ms = kernel["total_time_ns"] / 1_000_000

      print(
          f"{kernel['name']} | "
          f"{total_ms:.4f} ms | "
          f"{kernel['instances']} launches"
    )

    # For our candidate workload, the dominant kernel should be
    # the generated Triton kernel.
    top_kernel = kernels[0]

    print(f"\nProfiling generated kernel with NCU: {top_kernel['name']}")

    ncu_output = profile_kernel_with_ncu(
        workload,
        top_kernel["name"],
    )

    ncu_metrics = parse_ncu_output(ncu_output)

    return {
        "kernel": top_kernel,
        "metrics": ncu_metrics,
    }



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
Generate an improved, valid Triton implementation.

If a previous compiler/runtime error is provided, fix that error before
attempting further optimization.

Requirements:
- Preserve correctness.
- Use only valid Triton APIs.
- Wrapper MUST be named `triton_implementation`.
- Preserve the same input/output interface.
- Return Python code only.
- Use minimal comments.
"""


def optimize_triton_kernel(pytorch_code, triton_code, gpu_specs, benchmark, ncu_metrics, roofline,error=None):

    prompt = TRITON_OPTIMIZATION_PROMPT.format(
        gpu_specs=json.dumps(gpu_specs, indent=2),
        pytorch_code=pytorch_code,
        triton_code=triton_code,
        pytorch_ms=benchmark["pytorch_ms"],
        triton_ms=benchmark["triton_ms"],
        speedup=benchmark["speedup"],
        classification=roofline["classification"],
        compute_sol=roofline["compute_sol"],
        memory_sol=roofline["memory_sol"],
        efficiency=roofline["efficiency"],
        headroom=roofline["headroom"],
        ncu_metrics=json.dumps(ncu_metrics, indent=2),
        error=error or "None",
    )

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return extract_python_code(response.output_text)


if __name__ == "__main__":


 # ==========================================
    # TEST TRITON GENERATION
    # ==========================================

    gpu_specs = get_gpu_specs()

    pytorch_code = """
    def pytorch_reference(x):
        return torch.relu(x)
    """

    triton_code = generate_triton_kernel(
        pytorch_code,
        gpu_specs,
    )

    triton_code = extract_python_code(triton_code)

    with open("generated_kernel.py", "w") as f:
        f.write(triton_code)

    print("\n================ GENERATED TRITON ================")
    print(triton_code)

    print("\nGenerated candidate saved to generated_kernel.py")


    verification = verify_candidate()

    print("\n================ VERIFICATION ================")

    if verification["passed"]:
        print(f"PASS: {verification['tests']} tests passed")
    else:
        print("FAIL")
        print(f"Type: {verification['error_type']}")

        if "shape" in verification:
            print(f"Shape: {verification['shape']}")

        print(verification["error"])


    if verification["passed"]:
      benchmark = benchmark_candidate()

      print("\n================ BENCHMARK ================")
      print(f"PyTorch: {benchmark['pytorch_ms']:.4f} ms")
      print(f"Triton:  {benchmark['triton_ms']:.4f} ms")
      print(f"Speedup: {benchmark['speedup']:.2f}x")

    create_candidate_workload()

    candidate_profile = profile_candidate()

    if candidate_profile:
        candidate_metrics = candidate_profile["metrics"]

        print("\n================ CANDIDATE NCU METRICS ================")

        for name, value in candidate_metrics.items():
            print(f"{name}: {value}")

        candidate_roofline = analyze_roofline(candidate_metrics)

        print("\n================ CANDIDATE ROOFLINE ================")
        print(f"Classification: {candidate_roofline['classification']}")
        print(f"Compute SOL: {candidate_roofline['compute_sol']}%")
        print(f"Memory SOL: {candidate_roofline['memory_sol']}%")
        print(f"Efficiency: {candidate_roofline['efficiency']}%")
        print(f"Headroom: {candidate_roofline['headroom']}%")


        optimized_code = optimize_triton_kernel(
        pytorch_code=pytorch_code,
        triton_code=triton_code,
        gpu_specs=gpu_specs,
        benchmark=benchmark,
        ncu_metrics=candidate_metrics,
        roofline=candidate_roofline,
        )

    print("\n================ OPTIMIZED TRITON V2 ================")
    print(optimized_code)

    with open("generated_kernel_v2.py", "w") as f:
        f.write(optimized_code)

    print("\nOptimized candidate saved to generated_kernel_v2.py")

    v2_verification = verify_candidate("generated_kernel_v2.py")

    print("\n================ V2 VERIFICATION ================")

    if v2_verification["passed"]:
        print(f"PASS: {v2_verification['tests']} tests passed")

    else:
        print("FAIL")
        print(f"Type: {v2_verification['error_type']}")
        print(v2_verification["error"])

        optimized_code = optimize_triton_kernel(
            pytorch_code=pytorch_code,
            triton_code=optimized_code,
            gpu_specs=gpu_specs,
            benchmark=benchmark,
            ncu_metrics=candidate_metrics,
            roofline=candidate_roofline,
            error=v2_verification["error"],
        )

        with open("generated_kernel_v2.py", "w") as f:
            f.write(optimized_code)

        v2_verification = verify_candidate("generated_kernel_v2.py")

        print("\n================ V2 RETRY VERIFICATION ================")

        if v2_verification["passed"]:
            print(f"PASS: {v2_verification['tests']} tests passed")
        else:
            print("FAIL")
            print(v2_verification["error"])


    
    if v2_verification["passed"]:
        v2_benchmark = benchmark_candidate("generated_kernel_v2.py")

        print("\n================ V2 BENCHMARK ================")
        print(f"PyTorch:   {v2_benchmark['pytorch_ms']:.4f} ms")
        print(f"Triton V1: {benchmark['triton_ms']:.4f} ms")
        print(f"Triton V2: {v2_benchmark['triton_ms']:.4f} ms")
        print(f"V2 speedup vs PyTorch: {v2_benchmark['speedup']:.2f}x")

        if v2_benchmark["triton_ms"] < benchmark["triton_ms"]:
            print("\nV2 ACCEPTED: faster than V1")
        else:
            print("\nV2 REJECTED: V1 remains faster")



    sys.exit(0)





    if len(sys.argv) < 2:
        print("Usage: python gpuagent.py <workload>")
        sys.exit(1)

    # Run the workload under Nsight
    profile(sys.argv[1:])


    # Get raw Nsight reports
    kernel_stats = get_stats("cuda_gpu_kern_sum")
    cuda_api_stats = get_stats("cuda_api_sum")
    memory_stats = get_stats("cuda_gpu_mem_time_sum")
    nvtx_stats = get_stats("nvtx_sum")
    nvtx_kernel_stats = get_stats("nvtx_kern_sum")


    # Parse reports into Python data
    kernels = parse_kernel_stats(kernel_stats)
    cuda_calls = parse_cuda_api_stats(cuda_api_stats)
    memory_ops = parse_memory_stats(memory_stats)
    nvtx_ranges = parse_nvtx_stats(nvtx_stats)
    nvtx_kernels = parse_nvtx_kernel_stats(nvtx_kernel_stats)


    # Combine everything into one structured profile
    profile_data = {
        "kernels": kernels,
        "cuda_calls": cuda_calls,
        "memory_ops": memory_ops,
        "nvtx_ranges": nvtx_ranges,
        "nvtx_kernels": nvtx_kernels,
    }


    analyze_profile(profile_data)
    findings = diagnose_profile(profile_data)


    print("\n================ DIAGNOSTIC FINDINGS ================")

    for finding in findings:
        print(f"\n[{finding['severity'].upper()}] {finding['category']}")
        print(f"Evidence: {finding['evidence']}")
        print(f"Actionable: {finding['actionable']}")
        print(f"Reason: {finding['reason']}")




    if kernels:
        top_kernel = kernels[0]

        ncu_output = profile_kernel_with_ncu(
            sys.argv[1:],
            top_kernel["name"]
        )

        print("\n================ RAW NCU OUTPUT ================")
        print(ncu_output[:10000])

        ncu_metrics = parse_ncu_output(ncu_output)

        print("\n================ NCU KERNEL METRICS ================")
        for name, value in ncu_metrics.items():
            print(f"{name}: {value}")


        roofline = analyze_roofline(ncu_metrics)


        gpu_specs = get_gpu_specs()

        print("\n================ GPU SPECS ================")
        for name, value in gpu_specs.items():
            print(f"{name}: {value}")

        llm_diagnosis = analyze_with_llm(top_kernel["name"], ncu_metrics, roofline)

        print("\n================ LLM DIAGNOSIS ================")
        print(llm_diagnosis)



        print("\n================ ROOFLINE ANALYSIS ================")
        print(f"Classification: {roofline['classification']}")
        print(f"Compute SOL: {roofline['compute_sol']}%")
        print(f"Memory SOL: {roofline['memory_sol']}%")
        print(f"Efficiency: {roofline['efficiency']}%")
        print(f"At Roofline: {roofline['at_roofline']}")
        print(f"Headroom: {roofline['headroom']}%")

        kernel_diagnosis = diagnose_kernel(top_kernel, ncu_metrics)

        print("\n================ KERNEL DIAGNOSIS ================")
        print(f"Kernel: {kernel_diagnosis['kernel']}")
        print(f"Priority: {kernel_diagnosis['priority'].upper()}")

        print("\nEvidence:")
        for evidence in kernel_diagnosis["evidence"]:
            print(f"  - {evidence}")

        print(f"\nReason: {kernel_diagnosis['reason']}")
    else:
        print("\nNo GPU kernels were found, so NCU deep profiling was skipped.")



    # Raw reports
    #print("\n-------------------- KERNELS --------------------")
    #print(kernel_stats)

    #print("\n-------------------- CUDA API --------------------")
    #print(cuda_api_stats)

    #print("\n-------------------- MEMORY --------------------")
    #print(memory_stats)

    #print("\n-------------------- NVTX --------------------")
    #print(nvtx_stats)

    #print("\n-------------------- NVTX KERNELS --------------------")
    #print(nvtx_kernel_stats)

    # Structured data
    #print("\n-------------------- PROFILE DATA --------------------")
    #print(profile_data)
