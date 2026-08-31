iimport json
import subprocess


def verify_candidate(
    filename="generated_kernel.py",
    problem_file=None,
):
    if problem_file is None:
        return {
            "passed": False,
            "error_type": "execution",
            "error": "A problem file is required for verification.",
        }

    try:
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
            timeout=120,
        )

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error_type": "timeout",
            "error": "Verification exceeded 120 seconds.",
        }

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