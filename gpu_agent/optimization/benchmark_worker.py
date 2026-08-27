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

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(*inputs)
                candidate.triton_implementation(*inputs)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ---------------- PyTorch ----------------

        start.record()

        with torch.no_grad():
            for _ in range(20):
                model(*inputs)

        end.record()

        torch.cuda.synchronize()

        pytorch_ms = start.elapsed_time(end) / 20

        # ---------------- Triton ----------------

        start.record()

        for _ in range(20):
            candidate.triton_implementation(*inputs)

        end.record()

        torch.cuda.synchronize()

        triton_ms = start.elapsed_time(end) / 20

        print(json.dumps({
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "speedup": pytorch_ms / triton_ms,
        }))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()