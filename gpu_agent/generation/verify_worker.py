import importlib.util
import json
import sys

import torch


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        filename,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    candidate_file = sys.argv[1]
    problem_file = sys.argv[2]

    try:
        # Load generated Triton candidate
        candidate = load_module(
            candidate_file,
            "candidate_kernel",
        )

        # Load KernelBench problem
        problem = load_module(
            problem_file,
            "kernelbench_problem",
        )

        # Create reference PyTorch model
        model = problem.Model(
            *problem.get_init_inputs()
        ).cuda()

        model.eval()

        # Create KernelBench inputs
        inputs = problem.get_inputs()

        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        # Run reference and candidate
        with torch.no_grad():
            expected = model(*inputs)
            actual = candidate.triton_implementation(*inputs)

        # Check correctness
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

        print(json.dumps({
            "passed": True,
            "tests": 1,
        }))

    except AssertionError as e:
        print(json.dumps({
            "passed": False,
            "error_type": "correctness",
            "error": str(e),
        }))

    except Exception as e:
        print(json.dumps({
            "passed": False,
            "error_type": "execution",
            "error": str(e),
        }))


if __name__ == "__main__":
    main()