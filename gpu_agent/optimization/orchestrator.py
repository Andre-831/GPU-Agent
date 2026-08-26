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



def run_optimization(pytorch_code):
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


    verification = verify_candidate()

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


    benchmark = benchmark_candidate()

    print("\n================ BENCHMARK ================")
    print(f"PyTorch: {benchmark['pytorch_ms']:.4f} ms")
    print(f"Triton:  {benchmark['triton_ms']:.4f} ms")
    print(f"Speedup: {benchmark['speedup']:.2f}x")


    create_candidate_workload()
    candidate_profile = profile_candidate()

    if not candidate_profile:
        print("Candidate profiling failed.")
        return

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

    optimized_code = optimize_triton_kernel(
        pytorch_code=pytorch_code,
        triton_code=triton_code,
        gpu_specs=gpu_specs,
        benchmark=benchmark,
        ncu_metrics=candidate_metrics,
        roofline=candidate_roofline,
    )

    print("\n================ OPTIMIZED TRITON V2 ================")
    print(optimized_code)

    with open("generated_kernel_v2.py", "w") as f:
        f.write(optimized_code)

    print("\nOptimized candidate saved to generated_kernel_v2.py")



    v2_verification = verify_candidate("generated_kernel_v2.py")

    print("\n================ V2 VERIFICATION ================")

    if v2_verification["passed"]:
        print(f"PASS: {v2_verification['tests']} tests passed")

    else:
        print("FAIL")
        print(f"Type: {v2_verification['error_type']}")
        print(v2_verification["error"])

        # Give the LLM one correction attempt.
        optimized_code = optimize_triton_kernel(
            pytorch_code=pytorch_code,
            triton_code=optimized_code,
            gpu_specs=gpu_specs,
            benchmark=benchmark,
            ncu_metrics=candidate_metrics,
            roofline=candidate_roofline,
            error=v2_verification["error"],
        )

        with open("generated_kernel_v2.py", "w") as f:
            f.write(optimized_code)

        v2_verification = verify_candidate("generated_kernel_v2.py")

        print("\n================ V2 RETRY VERIFICATION ================")

        if v2_verification["passed"]:
            print(f"PASS: {v2_verification['tests']} tests passed")
        else:
            print("FAIL")
            print(v2_verification["error"])
            return

  
    # BENCHMARK V2
  

    v2_benchmark = benchmark_candidate("generated_kernel_v2.py")

    print("\n================ V2 BENCHMARK ================")
    print(f"PyTorch:   {v2_benchmark['pytorch_ms']:.4f} ms")
    print(f"Triton V1: {benchmark['triton_ms']:.4f} ms")
    print(f"Triton V2: {v2_benchmark['triton_ms']:.4f} ms")
    print(f"V2 speedup vs PyTorch: {v2_benchmark['speedup']:.2f}x")

    if v2_benchmark["triton_ms"] < benchmark["triton_ms"]:
        print("\nV2 ACCEPTED: faster than V1")
        return {
            "winner": "v2",
            "code": optimized_code,
            "benchmark": v2_benchmark,
        }
    else:
        print("\nV2 REJECTED: V1 remains faster")
        return {
            "winner": "v1",
            "code": triton_code,
            "benchmark": benchmark,
        }