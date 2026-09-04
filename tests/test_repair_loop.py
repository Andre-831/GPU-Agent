from gpu_agent.optimization import orchestrator


def test_failed_first_attempt_proceeds_to_repair(monkeypatch, tmp_path):
    initial_code = "class ModelNew: pass"
    repaired_code = "class ModelNew: repaired = True"
    verification_results = iter([
        {
            "passed": False,
            "error_type": "correctness",
            "error": "Output mismatch",
        },
        {
            "passed": True,
            "tests": 1,
        },
    ])
    repair_calls = []

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "create_openai_client", lambda: client)
    monkeypatch.setattr(
        orchestrator,
        "generate_triton_kernel",
        lambda pytorch_code, gpu_specs, client: initial_code,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_candidate",
        lambda filename, problem_file: next(verification_results),
    )

    def fake_repair_triton_kernel(**kwargs):
        repair_calls.append(kwargs)
        return repaired_code

    monkeypatch.setattr(
        orchestrator,
        "repair_triton_kernel",
        fake_repair_triton_kernel,
    )

    candidate = orchestrator.generate_verified_candidate(
        pytorch_code="reference code",
        gpu_specs={"name": "test GPU"},
        problem_file="problem.py",
        candidate_id=1,
    )

    assert len(repair_calls) == 1
    assert repair_calls[0]["refinement_history"] == [
        {
            "round": 1,
            "kernel_code": initial_code,
            "error_type": "correctness",
            "error": "Output mismatch",
        }
    ]
    assert candidate == {
        "id": 1,
        "code": repaired_code,
        "file": "generated_kernel_seed_1.py",
    }
    assert (tmp_path / "generated_kernel_seed_1.py").read_text() == repaired_code
    assert client.closed
