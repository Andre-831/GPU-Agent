# GPU Optimization — Agent Instructions

Follow the repository root AGENTS.md.

## Scope
- orchestrator.py: seed generation, verification, selection, iterative optimization, and final evaluation.
- benchmark.py and benchmark_worker.py: isolated timing and statistics.
- profiler.py: GPU specifications, Nsys/NCU collection, and parsing.
- roofline.py: throughput-based bottleneck heuristics.
- optimizer.py: LLM-guided kernel optimization.

## Pipeline Invariants
- Overlap independent seed LLM lifecycles, but serialize verification and benchmark GPU work on a single device.
- Isolate seed failures and preserve deterministic candidate ordering.
- Benchmark only verified candidates.
- Profile the CURRENT BEST implementation, not necessarily the latest candidate.
- Verify each optimized candidate before timing it.
- Retain the best verified kernel if a challenger fails or has worse measured median latency.
- Preserve final evaluation and report its failure independently of saving the winning file.
- Do not silently change seed counts, refinement limits, optimization rounds, or timeouts.

## Benchmarking
- Use real GPU measurements for performance conclusions.
- Preserve timing units, raw samples, median/mean/spread statistics, warmup and measurement budgets.
- Current selection uses Triton median latency; speedup is PyTorch median divided by Triton median.
- Preserve subprocess timeout/error handling and progress reporting.
- The current implementation does NOT remeasure the incumbent in paired trials; tiny median improvements may be noise. Do not claim statistical confidence without supporting tests.

## Profiling and Roofline
- Nsys describes execution-level activity; NCU supplies hardware metrics.
- Preserve metric names and units in parsers; handle unavailable metrics without fabrication.
- Do not confuse workload-wide timing with individual-kernel timing.
- Keep profiling overhead out of benchmark latency.
- roofline.py uses throughput thresholds as heuristics, not definitive proof of a bottleneck.
- When changing thresholds, metric parsing, or classifications, add focused regression tests.

## Tests
Cover candidate ordering and selection, failure isolation, verification gates, subprocess errors/timeouts, missing metrics, Roofline classifications, and preservation of the incumbent. Prefer mocks and synthetic metric data for unit tests; run CUDA integration checks separately.
