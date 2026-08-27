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


def evaluate(filename, problem_file):
    candidate = load_module(
        filename,
        "candidate_kernel",
    )

    problem = load_module(
        problem_file,
        "kernelbench_problem",
    )

    init_inputs = problem.get_init_inputs()

    model = problem.Model(*init_inputs)
    model = model.cuda()
    model.eval()


    for trial in range(N_CORRECTNESS):
        seed = trial

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = problem.get_inputs()

        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        with torch.no_grad():
            expected = model(*inputs)
            actual = candidate.triton_implementation(*inputs)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=RTOL,
            atol=ATOL,
        )

        del inputs
        del expected
        del actual

        torch.cuda.empty_cache()


    inputs = problem.get_inputs()

    inputs = [
        x.cuda() if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(*inputs)
            candidate.triton_implementation(*inputs)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # PyTorch timing
    start.record()

    with torch.no_grad():
        for _ in range(N_TRIALS):
            model(*inputs)

    end.record()
    torch.cuda.synchronize()

    pytorch_ms = start.elapsed_time(end) / N_TRIALS

    # Triton timing
    start.record()

    with torch.no_grad():
        for _ in range(N_TRIALS):
            candidate.triton_implementation(*inputs)

    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / N_TRIALS

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