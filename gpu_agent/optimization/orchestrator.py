from gpu_agent.generation.generator import (
    generate_triton_kernel,
    extract_python_code,
    repair_triton_kernel,
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


def generate_verified_candidate(
    pytorch_code,
    gpu_specs,
    problem_file,
    candidate_id,
    max_refinement_rounds=5,
):
    candidate_file = f"generated_kernel_seed_{candidate_id}.py"

    triton_code = generate_triton_kernel(
        pytorch_code,
        gpu_specs,
    )

    if triton_code is None:
        return None

    triton_code = extract_python_code(triton_code)

    if triton_code is None:
        return None

    refinement_history = []

    for refinement_round in range(1, max_refinement_rounds + 1):

        with open(candidate_file, "w") as f:
            f.write(triton_code)

        print(
            f"\n================ SEED {candidate_id} "
            f"GENERATION ROUND {refinement_round} ================"
        )
        print(triton_code)

        print(f"\nGenerated candidate saved to {candidate_file}")

        verification = verify_candidate(
            candidate_file,
            problem_file=problem_file,
        )

        print(
            f"\n================ SEED {candidate_id} "
            f"VERIFICATION ROUND {refinement_round} ================"
        )

        if verification["passed"]:
            print(
                f"PASS: {verification['tests']} tests passed"
            )

            return {
                "id": candidate_id,
                "code": triton_code,
                "file": candidate_file,
            }

        print("FAIL")
        print(f"Type: {verification['error_type']}")

        if "shape" in verification:
            print(f"Shape: {verification['shape']}")

        print(verification["error"])

        history_entry = {
            "round": refinement_round,
            "error_type": verification["error_type"],
            "error": verification["error"],
        }

        if "shape" in verification:
            history_entry["shape"] = verification["shape"]

        refinement_history.append(history_entry)

        if refinement_round == max_refinement_rounds:
            print(
                f"\nSeed {candidate_id} failed to generate a correct "
                f"kernel after {max_refinement_rounds} refinement rounds."
            )
            return None

        print("\nGiving LLM a correction attempt...")

        triton_code = repair_triton_kernel(
            pytorch_code=pytorch_code,
            triton_code=triton_code,
            gpu_specs=gpu_specs,
            error_type=verification["error_type"],
            error=verification["error"],
            refinement_history=refinement_history,
        )

        if triton_code is None:
            return None

    return None


def run_optimization(pytorch_code, problem_file=None):
    gpu_specs = get_gpu_specs()

    # Generate multiple independent starting candidates
    num_seeds = 4
    candidates = []

    for candidate_id in range(1, num_seeds + 1):

        print(
            f"\n================ SEED "
            f"{candidate_id}/{num_seeds} ================"
        )

        candidate = generate_verified_candidate(
            pytorch_code=pytorch_code,
            gpu_specs=gpu_specs,
            problem_file=problem_file,
            candidate_id=candidate_id,
        )

        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        print("\nNo seed produced a correct kernel.")
        return

    print(
        f"\n================ SEED RESULTS ================\n"
        f"{len(candidates)}/{num_seeds} seeds produced correct kernels."
    )

    # Benchmark every correct seed and select the fastest
    best_seed = None
    best_seed_benchmark = None

    for candidate in candidates:

        candidate_benchmark = benchmark_candidate(
            candidate["file"],
            problem_file=problem_file,
        )

        candidate["benchmark"] = candidate_benchmark

        print(
            f"\n================ SEED {candidate['id']} "
            f"BENCHMARK ================"
        )
        print(
            f"PyTorch: {candidate_benchmark['pytorch_ms']:.4f} ms"
        )
        print(
            f"Triton:  {candidate_benchmark['triton_ms']:.4f} ms"
        )
        print(
            f"Speedup: {candidate_benchmark['speedup']:.2f}x"
        )

        if (
            best_seed_benchmark is None
            or candidate_benchmark["triton_ms"]
            < best_seed_benchmark["triton_ms"]
        ):
            best_seed = candidate
            best_seed_benchmark = candidate_benchmark

    print(
        f"\n================ BEST SEED ================\n"
        f"Winner: Seed {best_seed['id']}"
    )
    print(
        f"PyTorch: {best_seed_benchmark['pytorch_ms']:.4f} ms"
    )
    print(
        f"Triton:  {best_seed_benchmark['triton_ms']:.4f} ms"
    )
    print(
        f"Speedup: {best_seed_benchmark['speedup']:.2f}x"
    )

    
    triton_code = best_seed["code"]
    benchmark = best_seed_benchmark

    with open("generated_kernel.py", "w") as f:
        f.write(triton_code)

    
    best_code = triton_code
    best_benchmark = benchmark
    best_version = 1

    max_itter = 5

    for iteration in range(2, max_itter + 2):

        print(
            f"\n================ OPTIMIZATION V{iteration} ================"
        )

        # Save current best as the kernel that will be profiled
        with open("generated_kernel.py", "w") as f:
            f.write(best_code)

        
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
        print(
            f"Classification: "
            f"{candidate_roofline['classification']}"
        )
        print(
            f"Compute SOL: "
            f"{candidate_roofline['compute_sol']}%"
        )
        print(
            f"Memory SOL: "
            f"{candidate_roofline['memory_sol']}%"
        )
        print(
            f"Efficiency: "
            f"{candidate_roofline['efficiency']}%"
        )
        print(
            f"Headroom: "
            f"{candidate_roofline['headroom']}%"
        )

        
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
            f"\n================ GENERATED TRITON "
            f"V{iteration} ================"
        )
        print(candidate_code)

        verification = verify_candidate(
            candidate_file,
            problem_file=problem_file,
        )

        print(
            f"\n================ V{iteration} "
            f"VERIFICATION ================"
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

            verification = verify_candidate(
                candidate_file,
                problem_file=problem_file,
            )

            print(
                f"\n================ V{iteration} "
                f"RETRY VERIFICATION ================"
            )

            if not verification["passed"]:
                print("FAIL")
                print(verification["error"])
                print(f"V{iteration} rejected.")
                continue

        print(
            f"PASS: {verification['tests']} tests passed"
        )

        candidate_benchmark = benchmark_candidate(
            candidate_file,
            problem_file=problem_file,
        )

        print(
            f"\n================ V{iteration} "
            f"BENCHMARK ================"
        )
        print(
            f"PyTorch:      "
            f"{candidate_benchmark['pytorch_ms']:.4f} ms"
        )
        print(
            f"Current best: "
            f"{best_benchmark['triton_ms']:.4f} ms"
        )
        print(
            f"Candidate:    "
            f"{candidate_benchmark['triton_ms']:.4f} ms"
        )
        print(
            f"Speedup:      "
            f"{candidate_benchmark['speedup']:.2f}x"
        )

        if (
            candidate_benchmark["triton_ms"]
            < best_benchmark["triton_ms"]
        ):

            print(
                f"\nV{iteration} ACCEPTED: "
                f"faster than V{best_version}"
            )

            best_code = candidate_code
            best_benchmark = candidate_benchmark
            best_version = iteration

        else:
            print(
                f"\nV{iteration} REJECTED: "
                f"V{best_version} remains faster"
            )

    with open("generated_kernel_best.py", "w") as f:
        f.write(best_code)

    print("\n================ OPTIMIZATION COMPLETE ================")
    print(f"Winner: V{best_version}")
    print(f"PyTorch: {best_benchmark['pytorch_ms']:.4f} ms")
    print(f"Triton:  {best_benchmark['triton_ms']:.4f} ms")
    print(f"Speedup: {best_benchmark['speedup']:.2f}x")
    print("Best kernel saved to generated_kernel_best.py")

    # Final evaluation
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
                f"PyTorch: "
                f"{final_evaluation['pytorch_ms']:.4f} ms"
            )

            print(
                f"Triton:  "
                f"{final_evaluation['triton_ms']:.4f} ms"
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
        "seed": best_seed["id"],
        "code": best_code,
        "benchmark": best_benchmark,
        "evaluation": final_evaluation,
    }