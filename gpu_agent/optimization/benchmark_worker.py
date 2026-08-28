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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        seed = 42

        # Get constructor arguments
        set_seed(seed)
        init_inputs = problem.get_init_inputs()

        # Create reference PyTorch model
        set_seed(seed)
        reference_model = problem.Model(
            *init_inputs
        ).cuda()

        # Create generated Triton model
        set_seed(seed)
        candidate_model = candidate.ModelNew(
            *init_inputs
        ).cuda()

        reference_model.eval()
        candidate_model.eval()

        # Create KernelBench inputs
        set_seed(seed)
        inputs = problem.get_inputs()

        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                reference_model(*inputs)
                candidate_model(*inputs)

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ---------------- PyTorch ----------------

        start.record()

        with torch.no_grad():
            for _ in range(20):
                reference_model(*inputs)

        end.record()

        torch.cuda.synchronize()

        pytorch_ms = start.elapsed_time(end) / 20

        # ---------------- Triton ----------------

        start.record()

        with torch.no_grad():
            for _ in range(20):
                candidate_model(*inputs)

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