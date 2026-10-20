# Testing — Agent Instructions

Follow the repository root AGENTS.md.

## Conventions
Use pytest and the existing tests/ organization:
- test_benchmark.py: timing and benchmark behavior.
- test_parallel_seeds.py: concurrent generation and orchestration.
- test_prompts.py: prompt and refinement-history behavior.
- test_repair_loop.py: repair/refinement behavior.

Run targeted tests first, then python -m pytest tests/ -q when feasible.

## Rules
- Prefer deterministic, focused tests with pytest fixtures and monkeypatch.
- Mock OpenAI, subprocesses, GPU tools, and profiling output in unit tests.
- Preserve conftest.py protection against real OpenAI API requests.
- Never require real credentials or network access for unit tests.
- Avoid brittle wall-clock timing assertions; use controlled samples.
- Cover both success and failure cases, including malformed output, timeouts, concurrency, and retention of the best verified candidate.
- Do not weaken assertions to make a failing implementation appear correct.

## GPU Validation
Keep CUDA/Triton/Nsys/NCU integration checks separate from hardware-independent unit tests. Do not assume an NVIDIA GPU is present locally. Clearly report which checks were mocked, which ran on real hardware, and which could not run.

## Bug Fix Workflow
When practical, first reproduce the bug in a regression test, implement the focused fix, run targeted tests, and then run the full available suite.
