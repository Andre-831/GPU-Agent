import subprocess
import csv
import io
import torch

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

def create_candidate_workload(problem_file=None):
    if problem_file is None:
        code = """
import torch
from generated_kernel import ModelNew

model = ModelNew().cuda()
model.eval()

x = torch.randn(10_000_000, device="cuda")

with torch.no_grad():
    for _ in range(5):
        y = model(x)

torch.cuda.synchronize()

with torch.no_grad():
    y = model(x)

torch.cuda.synchronize()
"""

    else:
        code = f"""
import importlib.util
import torch

from generated_kernel import ModelNew


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


spec = importlib.util.spec_from_file_location(
    "kernelbench_problem",
    {problem_file!r},
)

problem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problem)

seed = 42

set_seed(seed)
init_inputs = problem.get_init_inputs()

set_seed(seed)
model = ModelNew(*init_inputs).cuda()
model.eval()

set_seed(seed)
inputs = problem.get_inputs()

inputs = [
    x.cuda() if isinstance(x, torch.Tensor) else x
    for x in inputs
]

with torch.no_grad():
    for _ in range(5):
        y = model(*inputs)

torch.cuda.synchronize()

with torch.no_grad():
    y = model(*inputs)

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