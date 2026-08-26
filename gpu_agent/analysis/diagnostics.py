import os
import json
from openai import OpenAI

from gpu_agent.prompts import BOTTLENECK_PROMPT


client = OpenAI()

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

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return response.output_text
