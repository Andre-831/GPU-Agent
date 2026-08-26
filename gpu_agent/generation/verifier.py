import importlib.util
import torch


def verify_candidate(filename="generated_kernel.py"):
    try:
        spec = importlib.util.spec_from_file_location(
            "candidate_kernel",
            filename
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_shapes = [
            (1024,),
            (4096,),
            (256, 256),
            (12345,),
        ]

        for shape in test_shapes:
            x = torch.randn(shape, device="cuda")

            expected = torch.relu(x)
            actual = module.triton_implementation(x)

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