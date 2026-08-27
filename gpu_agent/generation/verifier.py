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


def verify_candidate(
    filename="generated_kernel.py",
    problem_file=None,
):


    if problem_file is not None:
        result = subprocess.run(
            [
                "python",
                "-m",
                "gpu_agent.generation.verify_worker",
                filename,
                problem_file,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "passed": False,
                "error_type": "execution",
                "error": result.stderr,
            }

        try:
            return json.loads(result.stdout.strip())

        except json.JSONDecodeError:
            return {
                "passed": False,
                "error_type": "execution",
                "error": (
                    "Could not parse verification worker output:\n"
                    + result.stdout
                ),
            }

    # OLD RELU VERIFICATION
    # Keep this so the original standalone test still works.


    try:
        candidate = load_module(
            filename,
            "candidate_kernel",
        )

        test_shapes = [
            (1024,),
            (4096,),
            (256, 256),
            (12345,),
        ]

        for shape in test_shapes:
            x = torch.randn(
                shape,
                device="cuda",
            )

            expected = torch.relu(x)
            actual = candidate.triton_implementation(x)

            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )

            except AssertionError as e:
                return {
                    "passed": False,
                    "error_type": "correctness",
                    "shape": shape,
                    "error": str(e),
                }

        return {
            "passed": True,
            "tests": len(test_shapes),
        }

    except Exception as e:
        return {
            "passed": False,
            "error_type": "execution",
            "error": str(e),
        }