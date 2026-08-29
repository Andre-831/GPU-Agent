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


def tensors_allclose(actual, expected, rtol=1e-4, atol=1e-4):
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)

    chunk_size = 10_000_000

    for start in range(0, actual_flat.numel(), chunk_size):
        end = min(
            start + chunk_size,
            actual_flat.numel(),
        )

        actual_chunk = actual_flat[start:end]
        expected_chunk = expected_flat[start:end]

        if not torch.allclose(
            actual_chunk,
            expected_chunk,
            rtol=rtol,
            atol=atol,
        ):
            # Calculate diagnostics only for the failing chunk
            # to avoid large full-tensor temporary allocations.
            diff = torch.abs(
                actual_chunk - expected_chunk
            )

            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Find the element with the largest error.
            local_index = diff.argmax().item()
            global_index = start + local_index

            actual_value = actual_chunk[local_index].item()
            expected_value = expected_chunk[local_index].item()

            return False, {
                "chunk_start": start,
                "chunk_end": end,
                "index": global_index,
                "actual": actual_value,
                "expected": expected_value,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
            }

    return True, None


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

        # Create identical KernelBench inputs
        set_seed(seed)
        inputs = problem.get_inputs()

        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        # Run reference and candidate
        with torch.no_grad():
            expected = reference_model(*inputs)
            actual = candidate_model(*inputs)

        # Check output shape
        if expected.shape != actual.shape:
            raise AssertionError(
                f"Shape mismatch: expected {expected.shape}, "
                f"got {actual.shape}"
            )

        # Compare outputs in chunks to avoid OOM
        close, diagnostics = tensors_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

        if not close:
            raise AssertionError(
                "Output mismatch. "
                f"First failing chunk: "
                f"{diagnostics['chunk_start']}:"
                f"{diagnostics['chunk_end']}. "
                f"Largest mismatch at flattened index "
                f"{diagnostics['index']}. "
                f"Expected {diagnostics['expected']}, "
                f"got {diagnostics['actual']}. "
                f"Max absolute difference: "
                f"{diagnostics['max_abs_diff']}. "
                f"Mean absolute difference in chunk: "
                f"{diagnostics['mean_abs_diff']}."
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