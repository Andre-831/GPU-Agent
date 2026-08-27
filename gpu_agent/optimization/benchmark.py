import importlib.util
import json
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


def benchmark_candidate(
    filename="generated_kernel.py",
    problem_file=None,
):
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
            return json.loads(result.stdout.strip())

        except json.JSONDecodeError:
            raise RuntimeError(
                "Could not parse benchmark worker output:\n"
                + result.stdout
            )
            
    # OLD RELU TEST
    candidate = load_module(
        filename,
        "candidate_kernel",
    )

    x = torch.randn(
        10_000_000,
        device="cuda",
    )

    # Warmup
    for _ in range(10):
        torch.relu(x)
        candidate.triton_implementation(x)

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
        candidate.triton_implementation(x)

    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / 100

    return {
        "pytorch_ms": pytorch_ms,
        "triton_ms": triton_ms,
        "speedup": pytorch_ms / triton_ms,
    }