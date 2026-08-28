import argparse
import importlib.util
import json
import random

import torch


N_CORRECTNESS = 5
N_TRIALS = 20
RTOL = 1e-4
ATOL = 1e-4


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        filename,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensors_allclose(actual, expected, rtol=1e-4, atol=1e-4):
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)

    # Compare large outputs in chunks to avoid temporary
    # allocations from torch.allclose causing GPU OOM.
    chunk_size = 10_000_000

    for start in range(0, actual_flat.numel(), chunk_size):
        end = min(
            start + chunk_size,
            actual_flat.numel(),
        )

        if not torch.allclose(
            actual_flat[start:end],
            expected_flat[start:end],
            rtol=rtol,
            atol=atol,
        ):
            return False

    return True


def evaluate(filename, problem_file):
    candidate = load_module(
        filename,
        "candidate_kernel",
    )

    problem = load_module(
        problem_file,
        "kernelbench_problem",
    )

    # Use the same seed when constructing both models so that
    # parameterized models initialize equivalent state.
    model_seed = 42

    set_seed(model_seed)
    init_inputs = problem.get_init_inputs()

    set_seed(model_seed)
    reference_model = problem.Model(
        *init_inputs
    ).cuda()

    set_seed(model_seed)
    candidate_model = candidate.ModelNew(
        *init_inputs
    ).cuda()

    reference_model.eval()
    candidate_model.eval()

    # ---------------- Correctness ----------------

    for trial in range(N_CORRECTNESS):
        seed = trial

        set_seed(seed)
        inputs = problem.get_inputs()

        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        with torch.no_grad():
            expected = reference_model(*inputs)
            actual = candidate_model(*inputs)

        if expected.shape != actual.shape:
            raise AssertionError(
                f"Shape mismatch on trial {trial}: "
                f"expected {expected.shape}, "
                f"got {actual.shape}"
            )

        if not tensors_allclose(
            actual,
            expected,
            rtol=RTOL,
            atol=ATOL,
        ):
            raise AssertionError(
                f"Output mismatch on trial {trial}"
            )

        del inputs
        del expected
        del actual

        torch.cuda.empty_cache()

    # ---------------- Performance ----------------

    set_seed(100)
    inputs = problem.get_inputs()

    inputs = [
        x.cuda() if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            reference_model(*inputs)
            candidate_model(*inputs)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ---------------- PyTorch timing ----------------

    start.record()

    with torch.no_grad():
        for _ in range(N_TRIALS):
            reference_model(*inputs)

    end.record()

    torch.cuda.synchronize()

    pytorch_ms = (
        start.elapsed_time(end) / N_TRIALS
    )

    # ---------------- Triton timing ----------------

    start.record()

    with torch.no_grad():
        for _ in range(N_TRIALS):
            candidate_model(*inputs)

    end.record()

    torch.cuda.synchronize()

    triton_ms = (
        start.elapsed_time(end) / N_TRIALS
    )

    speedup = pytorch_ms / triton_ms

    return {
        "correct": True,
        "correctness_tests": N_CORRECTNESS,
        "pytorch_ms": pytorch_ms,
        "triton_ms": triton_ms,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        required=True,
    )

    parser.add_argument(
        "--problem-file",
        required=True,
    )

    args = parser.parse_args()

    try:
        result = evaluate(
            args.filename,
            args.problem_file,
        )

    except AssertionError as e:
        result = {
            "correct": False,
            "error_type": "correctness",
            "error": str(e),
        }

    except Exception as e:
        result = {
            "correct": False,
            "error_type": "execution",
            "error": str(e),
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()