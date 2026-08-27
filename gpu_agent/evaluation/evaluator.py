import json
import subprocess
import sys


def evaluate_kernel(
    filename="generated_kernel_best.py",
    problem_file=None,
):
    if problem_file is None:
        return {
            "correct": False,
            "error_type": "configuration",
            "error": "problem_file is required for final evaluation",
        }

    command = [
        sys.executable,
        "-m",
        "gpu_agent.evaluation.eval_worker",
        "--filename",
        filename,
        "--problem-file",
        problem_file,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return {
                "correct": False,
                "error_type": "worker",
                "error": result.stderr.strip()
                or "Evaluation worker failed",
            }

        # Worker should print JSON as its final line.
        output_lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        if not output_lines:
            return {
                "correct": False,
                "error_type": "worker",
                "error": "Evaluation worker returned no output",
            }

        return json.loads(output_lines[-1])

    except json.JSONDecodeError as e:
        return {
            "correct": False,
            "error_type": "parsing",
            "error": f"Could not parse evaluation result: {e}",
        }

    except Exception as e:
        return {
            "correct": False,
            "error_type": "execution",
            "error": str(e),
        }