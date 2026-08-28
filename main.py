import sys

from gpu_agent.optimization.orchestrator import run_optimization


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <kernelbench_problem.py>")
        sys.exit(1)

    problem_file = sys.argv[1]

    with open(problem_file, "r") as f:
        pytorch_code = f.read()

    run_optimization(
        pytorch_code,
        problem_file=problem_file,
    )


if __name__ == "__main__":
    main()