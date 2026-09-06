import importlib.util
import json
import math
import statistics
import subprocess
import sys

import torch
from triton.testing import do_bench


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


def validate_budgets(warmup_ms, rep_ms):
    for name, value, allow_zero in (
        ("warmup_ms", warmup_ms, True),
        ("rep_ms", rep_ms, False),
    ):
        validate_duration(name, value, allow_zero=allow_zero)


def validate_duration(name, value, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (value == 0 and not allow_zero)):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be a finite number {relation}")


def benchmark_progress(message):
    print(f"[Benchmark] {message}", file=sys.stderr, flush=True)


def benchmark_functions(pytorch_call, triton_call, warmup_ms=25, rep_ms=100):
    """Collect adaptive per-call samples using approximate millisecond budgets.

    Triton performs initial/calibration calls in addition to these budgets.
    A single sample's zero stddev does not establish measurement certainty.
    """
    validate_budgets(warmup_ms, rep_ms)
    with torch.no_grad():
        timings = {}
        for name, label, call in (
            ("pytorch", "PyTorch", pytorch_call),
            ("triton", "Triton", triton_call),
        ):
            benchmark_progress(
                f"{label} timing (warmup={warmup_ms} ms, measurement={rep_ms} ms)..."
            )
            samples = do_bench(call, warmup=warmup_ms, rep=rep_ms, return_mode="all")
            timings[f"{name}_stats"] = timing_statistics(samples)
            benchmark_progress(f"{label} timing finished ({len(samples)} samples).")
        benchmark_progress("Timing complete.")

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
    warmup_ms=25,
    rep_ms=100,
    timeout_s=300,
):
    """Benchmark with millisecond budgets and a worker wall-clock timeout.

    Worker stdout is JSON; stderr is inherited for live diagnostics. A timeout
    kills/reaps the worker and raises RuntimeError. The legacy ReLU path runs
    in-process, so timeout_s only applies when problem_file is provided.
    """
    validate_budgets(warmup_ms, rep_ms)
    validate_duration("timeout_s", timeout_s)
    # ============================================================
    # KERNELBENCH
    # Run benchmark in its own process so GPU memory is released
    # when the benchmark finishes.
    # ============================================================

    if problem_file is not None:
        benchmark_progress(f"Setup: starting worker for {filename} (timeout={timeout_s} s)...")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gpu_agent.optimization.benchmark_worker",
                    filename,
                    problem_file,
                    str(warmup_ms),
                    str(rep_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=None,  # Inherit stderr so progress is visible immediately.
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Benchmark worker timed out after {timeout_s} seconds for {filename}"
            ) from error

        if result.returncode != 0:
            raise RuntimeError(
                "Benchmark worker failed:\n"
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

    # OLD RELU TEST (in-process; subprocess timeout does not apply)
    benchmark_progress("Setup: loading ReLU benchmark...")
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
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )
    print_timing_summary(benchmark)
    return benchmark
