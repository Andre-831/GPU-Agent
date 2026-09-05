import importlib.util
import json
import statistics
import subprocess

import torch


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        filename,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def timing_statistics(samples_ms):
    """Summarize per-call timings in milliseconds."""
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "stddev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "num_samples": len(samples_ms),
        "samples_ms": list(samples_ms),
    }


def validate_counts(warmup_count, sample_count):
    for name, value, minimum in (
        ("warmup_count", warmup_count, 0),
        ("sample_count", sample_count, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def benchmark_functions(pytorch_call, triton_call, warmup_count=10, sample_count=30):
    """Warm both implementations, then time separate batches of 20 calls."""
    validate_counts(warmup_count, sample_count)
    with torch.no_grad():
        for _ in range(warmup_count):
            pytorch_call()
            triton_call()
        torch.cuda.synchronize()

        timings = {}
        for name, call in (("pytorch", pytorch_call), ("triton", triton_call)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(sample_count):
                start.record()
                for _ in range(20):
                    call()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / 20)
            timings[f"{name}_stats"] = timing_statistics(samples)

    pytorch_ms = timings["pytorch_stats"]["median_ms"]
    triton_ms = timings["triton_stats"]["median_ms"]
    return {
        "pytorch_ms": pytorch_ms,
        "triton_ms": triton_ms,
        "speedup": pytorch_ms / triton_ms,
        **timings,
    }


def print_timing_summary(result):
    for name, label in (("pytorch", "PyTorch"), ("triton", "Triton")):
        stats = result[f"{name}_stats"]
        print(
            f"{label}: median {stats['median_ms']:.6f} ms, "
            f"mean {stats['mean_ms']:.6f}, stddev {stats['stddev_ms']:.6f}, "
            f"range [{stats['min_ms']:.6f}, {stats['max_ms']:.6f}] "
            f"({stats['num_samples']} samples)"
        )


def benchmark_candidate(
    filename="generated_kernel.py",
    problem_file=None,
    warmup_count=10,
    sample_count=30,
):
    validate_counts(warmup_count, sample_count)
    # ============================================================
    # KERNELBENCH
    # Run benchmark in its own process so GPU memory is released
    # when the benchmark finishes.
    # ============================================================

    if problem_file is not None:
        result = subprocess.run(
            [
                "python",
                "-m",
                "gpu_agent.optimization.benchmark_worker",
                filename,
                problem_file,
                str(warmup_count),
                str(sample_count),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Benchmark worker failed:\n"
                + result.stderr
                + "\n"
                + result.stdout
            )

        try:
            benchmark = json.loads(result.stdout.strip())

        except json.JSONDecodeError:
            raise RuntimeError(
                "Could not parse benchmark worker output:\n"
                + result.stdout
            )

        print_timing_summary(benchmark)
        return benchmark

    # OLD RELU TEST
    candidate = load_module(
        filename,
        "candidate_kernel",
    )

    x = torch.randn(
        10_000_000,
        device="cuda",
    )

    benchmark = benchmark_functions(
        lambda: torch.relu(x),
        lambda: candidate.triton_implementation(x),
        warmup_count=warmup_count,
        sample_count=sample_count,
    )
    print_timing_summary(benchmark)
    return benchmark
