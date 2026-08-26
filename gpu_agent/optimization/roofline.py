def analyze_roofline(metrics):
    compute_sol = metrics.get(
      "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )

    memory_sol = metrics.get(
      "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    )

    if compute_sol is None or memory_sol is None:
      return {
        "classification": "unknown",
        "compute_sol": compute_sol,
        "memory_sol": memory_sol,
        "efficiency": None,
        "at_roofline": False,
        "headroom": None,
      }

    if compute_sol < 60 and memory_sol < 60:
        classification = "underutilized"
    elif memory_sol >= compute_sol:
        classification = "memory_bound"
    else:
        classification = "compute_bound"

    efficiency = max(compute_sol, memory_sol)

    return {
        "classification": classification,
        "compute_sol": compute_sol,
        "memory_sol": memory_sol,
        "efficiency": efficiency,
        "at_roofline": efficiency >= 95,
        "headroom": max(0, 100 - efficiency),
    }
