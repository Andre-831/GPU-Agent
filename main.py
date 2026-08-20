%%writefile gpuagent.py

import subprocess
import sys
import csv
import io


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

def profile_kernel_with_ncu(workload, kernel_name):
    command = [
        "ncu",
        "--csv",
        "--set", "basic",
        "--kernel-name", kernel_name,
        "--launch-count", "1",
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
    metrics = {}

    lines = output.splitlines()

    # Find the actual NCU CSV header
    for i, line in enumerate(lines):
        if '"Metric Name"' in line and '"Metric Value"' in line:
            csv_text = "\n".join(lines[i:])
            break
    else:
        return metrics

    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")

        if not name or not value:
            continue

        value = value.replace(",", "")

        try:
            value = float(value)
        except ValueError:
            continue

        metrics[name] = value

    return {
    "duration_ns": metrics.get("Duration"),
    "compute_throughput": metrics.get("Compute (SM) Throughput"),
    "memory_throughput": metrics.get("Memory Throughput"),
    "dram_throughput": metrics.get("DRAM Throughput"),
    "registers_per_thread": metrics.get("Registers Per Thread"),
    "waves_per_sm": metrics.get("Waves Per SM"),
    "theoretical_occupancy": metrics.get("Theoretical Occupancy"),
    "achieved_occupancy": metrics.get("Achieved Occupancy"),
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



if __name__ == "__main__":

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



  
    ncu_output = profile_kernel_with_ncu(
    sys.argv[1:],
    kernels[0]["name"]
    )

    ncu_metrics = parse_ncu_output(ncu_output)
    print(ncu_metrics)



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
