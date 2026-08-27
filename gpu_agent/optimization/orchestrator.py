from gpu_agent.generation.generator import (
    generate_triton_kernel,
    extract_python_code,
)
from gpu_agent.generation.verifier import verify_candidate

from gpu_agent.optimization.benchmark import benchmark_candidate
from gpu_agent.optimization.profiler import (
    get_gpu_specs,
    create_candidate_workload,
    profile_candidate,
)
from gpu_agent.optimization.roofline import analyze_roofline
from gpu_agent.optimization.optimizer import optimize_triton_kernel
from gpu_agent.evaluation.evaluator import evaluate_kernel



def run_optimization(pytorch_code, problem_file=None):
    gpu_specs = get_gpu_specs()


    triton_code = generate_triton_kernel(
        pytorch_code,
        gpu_specs,
    )

    triton_code = extract_python_code(triton_code)

    with open("generated_kernel.py", "w") as f:
        f.write(triton_code)

    print("\n================ GENERATED TRITON ================")
    print(triton_code)

    print("\nGenerated candidate saved to generated_kernel.py")


    verification = verify_candidate( problem_file=problem_file)

    print("\n================ VERIFICATION ================")

    if verification["passed"]:
        print(f"PASS: {verification['tests']} tests passed")
    else:
        print("FAIL")
        print(f"Type: {verification['error_type']}")

        if "shape" in verification:
            print(f"Shape: {verification['shape']}")

        print(verification["error"])
        return


    benchmark = benchmark_candidate(problem_file=problem_file)

    print("\n================ BENCHMARK ================")
    print(f"PyTorch: {benchmark['pytorch_ms']:.4f} ms")
    print(f"Triton:  {benchmark['triton_ms']:.4f} ms")
    print(f"Speedup: {benchmark['speedup']:.2f}x")



    # iterative loop optimization

    best_code = triton_code
    best_benchmark = benchmark
    best_version = 1

    max_itter = 5

    for iteration in range(2, max_itter + 2):

        print(
            f"\n================ OPTIMIZATION V{iteration} ================"
        )

        # save cur best as the kernel that will be profiled
        with open("generated_kernel.py", "w") as f:
            f.write(best_code)

        # Profile current best
        create_candidate_workload(problem_file=problem_file)
        candidate_profile = profile_candidate()

        if not candidate_profile:
            print("Candidate profiling failed.")
            break

        candidate_metrics = candidate_profile["metrics"]

        print("\n================ CANDIDATE NCU METRICS ================")

        for name, value in candidate_metrics.items():
            print(f"{name}: {value}")

        candidate_roofline = analyze_roofline(candidate_metrics)

        print("\n================ CANDIDATE ROOFLINE ================")
        print(f"Classification: {candidate_roofline['classification']}")
        print(f"Compute SOL: {candidate_roofline['compute_sol']}%")
        print(f"Memory SOL: {candidate_roofline['memory_sol']}%")
        print(f"Efficiency: {candidate_roofline['efficiency']}%")
        print(f"Headroom: {candidate_roofline['headroom']}%")

        # generate an optimization of the cur best
        candidate_code = optimize_triton_kernel(
            pytorch_code=pytorch_code,
            triton_code=best_code,
            gpu_specs=gpu_specs,
            benchmark=best_benchmark,
            ncu_metrics=candidate_metrics,
            roofline=candidate_roofline,
        )

        candidate_code = extract_python_code(candidate_code)

        candidate_file = f"generated_kernel_v{iteration}.py"

        with open(candidate_file, "w") as f:
            f.write(candidate_code)

        print(
            f"\n================ GENERATED TRITON V{iteration} ================"
        )
        print(candidate_code)

        verification = verify_candidate(candidate_file,problem_file=problem_file)

        print(
            f"\n================ V{iteration} VERIFICATION ================"
        )

        if not verification["passed"]:
            print("FAIL")
            print(f"Type: {verification['error_type']}")
            print(verification["error"])

            print("\nGiving LLM one correction attempt...")

            candidate_code = optimize_triton_kernel(
                pytorch_code=pytorch_code,
                triton_code=candidate_code,
                gpu_specs=gpu_specs,
                benchmark=best_benchmark,
                ncu_metrics=candidate_metrics,
                roofline=candidate_roofline,
                error=verification["error"],
            )

            candidate_code = extract_python_code(candidate_code)

            with open(candidate_file, "w") as f:
                f.write(candidate_code)

            verification = verify_candidate(candidate_file, problem_file=problem_file)

            print(
                f"\n================ V{iteration} RETRY VERIFICATION ================"
            )

            if not verification["passed"]:
                print("FAIL")
                print(verification["error"])
                print(f"V{iteration} rejected.")
                continue

        print(f"PASS: {verification['tests']} tests passed")

        candidate_benchmark = benchmark_candidate(candidate_file, problem_file=problem_file)

        print(
            f"\n================ V{iteration} BENCHMARK ================"
        )
        print(f"PyTorch:      {candidate_benchmark['pytorch_ms']:.4f} ms")
        print(f"Current best: {best_benchmark['triton_ms']:.4f} ms")
        print(f"Candidate:    {candidate_benchmark['triton_ms']:.4f} ms")
        print(f"Speedup:      {candidate_benchmark['speedup']:.2f}x")

        if candidate_benchmark["triton_ms"] < best_benchmark["triton_ms"]:

            print(
                f"\nV{iteration} ACCEPTED: faster than V{best_version}"
            )

            best_code = candidate_code
            best_benchmark = candidate_benchmark
            best_version = iteration

        else:
            print(
                f"\nV{iteration} REJECTED: V{best_version} remains faster"
            )

    with open("generated_kernel_best.py", "w") as f:
        f.write(best_code)

    print("\n================ OPTIMIZATION COMPLETE ================")
    print(f"Winner: V{best_version}")
    print(f"PyTorch: {best_benchmark['pytorch_ms']:.4f} ms")
    print(f"Triton:  {best_benchmark['triton_ms']:.4f} ms")
    print(f"Speedup: {best_benchmark['speedup']:.2f}x")
    print("Best kernel saved to generated_kernel_best.py")

    # FINAL EVALUATION

    if problem_file is not None:

        print("\n================ FINAL EVALUATION ================")

        final_evaluation = evaluate_kernel(
            filename="generated_kernel_best.py",
            problem_file=problem_file,
        )

        if final_evaluation["correct"]:

            print(
                f"Correctness: PASS "
                f"({final_evaluation['correctness_tests']} trials)"
            )

            print(
                f"PyTorch: {final_evaluation['pytorch_ms']:.4f} ms"
            )

            print(
                f"Triton:  {final_evaluation['triton_ms']:.4f} ms"
            )

            print(
                f"Final speedup: "
                f"{final_evaluation['speedup']:.2f}x"
            )

        else:

            print("FINAL EVALUATION FAILED")

            print(
                f"Type: "
                f"{final_evaluation.get('error_type', 'unknown')}"
            )

            print(
                final_evaluation.get(
                    "error",
                    "Unknown evaluation error",
                )
            )

    else:
        final_evaluation = None


    return {
        "winner": f"v{best_version}",
        "code": best_code,
        "benchmark": best_benchmark,
        "evaluation": final_evaluation,
    }