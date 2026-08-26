import importlib.util
import torch


def benchmark_candidate(filename="generated_kernel.py"):
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