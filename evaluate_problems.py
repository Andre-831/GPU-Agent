from gpu_agent.optimization.orchestrator import run_optimization


from pathlib import Path

LEVEL1_DIR = Path("/content/KernelBench/KernelBench/level1")

PROBLEMS = sorted(
    LEVEL1_DIR.glob("*.py"),
    key=lambda path: int(path.name.split("_")[0]),
)

results = []

for problem in PROBLEMS:
    print(f"\nEvaluating {problem}")

    try:
        with open(problem, "r") as f:
            pytorch_code = f.read()

        result = run_optimization(
            pytorch_code,
            problem_file=str(problem),
        )

        results.append({
            "problem": str(problem),
            "winner": result["winner"],
            "seed": result["seed"],
            "correct": result["evaluation"]["correct"],
            "pytorch_ms": result["evaluation"]["pytorch_ms"],
            "triton_ms": result["evaluation"]["triton_ms"],
            "speedup": result["evaluation"]["speedup"],
        })

    except Exception as e:
        results.append({
            "problem": problem,
            "error": str(e),
        })


print("\n================ SUMMARY ================")

for result in results:
    print(result)
