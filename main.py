from gpu_agent.optimization.orchestrator import run_optimization


def main():
    problem_file = (
        "external/KernelBench/KernelBench/level1/"
        "47_Sum_reduction_over_a_dimension.py"
    )

    # Give the generator the actual KernelBench problem source
    with open(problem_file, "r") as f:
        pytorch_code = f.read()

    run_optimization(
        pytorch_code,
        problem_file=problem_file,
    )


if __name__ == "__main__":
    main()