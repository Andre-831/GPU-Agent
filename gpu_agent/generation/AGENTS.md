# Kernel Generation — Agent Instructions

Follow the repository root AGENTS.md.

## Scope
- generator.py: OpenAI requests, Triton code extraction, initial generation, and repair.
- verifier.py: isolated correctness-verification subprocess and timeout.
- verify_worker.py: KernelBench model loading and PyTorch-versus-Triton comparison.
- ../prompts.py: shared generation and precision instructions.

## Rules
- Preserve PyTorch reference semantics and the KernelBench ModelNew interface.
- Keep generation and repair separate from verification.
- Use shared prompts and retain GPU specifications, precision guidance, and meaningful refinement history.
- Handle empty, malformed, and fenced LLM responses safely.
- Repair based on actual verification errors; bound attempts and do not repeatedly resubmit identical failures without reason.
- Never infer correctness from an LLM response alone.

## Verification
- Preserve deterministic initialization and input seeding.
- Check output shape and numerical agreement; existing chunked allclose checks use rtol=1e-4 and atol=1e-4.
- Preserve chunked comparisons for large tensors to limit memory overhead.
- Preserve subprocess isolation, verification timeouts, and structured failure details.
- Keep the orchestrator's single-GPU verification serialization.
- Generated kernels are untrusted code: never bypass verification or execute outside the intended verification/evaluation flow.

## Tests
Mock API calls. Cover code extraction, prompt/refinement-history formatting, valid/invalid candidates, numerical mismatches, execution failures, and timeouts. Only claim real GPU correctness after hardware verification.
