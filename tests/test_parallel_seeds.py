import threading
import time

import pytest

from gpu_agent.optimization import orchestrator


def test_seed_lifecycles_run_concurrently(monkeypatch):
    release_workers = threading.Event()
    all_workers_started = threading.Event()
    state_lock = threading.Lock()
    active_workers = 0
    maximum_active_workers = 0

    def fake_generate_verified_candidate(**kwargs):
        nonlocal active_workers, maximum_active_workers

        with state_lock:
            active_workers += 1
            maximum_active_workers = max(maximum_active_workers, active_workers)
            if active_workers == 4:
                all_workers_started.set()

        assert release_workers.wait(timeout=2)

        with state_lock:
            active_workers -= 1

        candidate_id = kwargs["candidate_id"]
        return {"id": candidate_id}

    monkeypatch.setattr(
        orchestrator,
        "generate_verified_candidate",
        fake_generate_verified_candidate,
    )

    def release_when_all_started():
        all_workers_started.wait(timeout=2)
        release_workers.set()

    release_thread = threading.Thread(target=release_when_all_started)
    release_thread.start()
    try:
        candidates = orchestrator.generate_seed_candidates(
            pytorch_code="reference",
            gpu_specs={},
            problem_file="problem.py",
        )
    finally:
        release_workers.set()
        release_thread.join(timeout=2)

    assert all_workers_started.is_set()
    assert maximum_active_workers == 4
    assert [candidate["id"] for candidate in candidates] == [1, 2, 3, 4]


def test_seed_failure_is_isolated_and_results_are_ordered(monkeypatch):
    completion_delays = {1: 0.04, 2: 0.03, 3: 0.02, 4: 0.01}

    def fake_generate_verified_candidate(**kwargs):
        candidate_id = kwargs["candidate_id"]
        time.sleep(completion_delays[candidate_id])

        if candidate_id == 2:
            raise RuntimeError("seed failed")

        if candidate_id == 3:
            return None

        return {"id": candidate_id}

    monkeypatch.setattr(
        orchestrator,
        "generate_verified_candidate",
        fake_generate_verified_candidate,
    )

    candidates = orchestrator.generate_seed_candidates(
        pytorch_code="reference",
        gpu_specs={},
        problem_file="problem.py",
    )

    assert [candidate["id"] for candidate in candidates] == [1, 4]


def test_seed_exception_logs_traceback(monkeypatch, capsys):
    def fake_generate_verified_candidate(**kwargs):
        if kwargs["candidate_id"] == 1:
            raise RuntimeError("unexpected failure")
        return None

    monkeypatch.setattr(
        orchestrator,
        "generate_verified_candidate",
        fake_generate_verified_candidate,
    )

    orchestrator.generate_seed_candidates(
        pytorch_code="reference",
        gpu_specs={},
        problem_file="problem.py",
    )

    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" in captured.err
    assert "RuntimeError: unexpected failure" in captured.err


@pytest.mark.parametrize("failure_stage", ["generation", "verification", "repair"])
def test_client_closes_when_seed_lifecycle_raises(
    monkeypatch,
    tmp_path,
    failure_stage,
):
    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "create_openai_client", lambda: client)

    def fake_generate(*args, **kwargs):
        if failure_stage == "generation":
            raise RuntimeError("generation failed")
        return "class ModelNew: pass"

    def fake_verify(*args, **kwargs):
        if failure_stage == "verification":
            raise RuntimeError("verification failed")
        return {
            "passed": False,
            "error_type": "correctness",
            "error": "Output mismatch",
        }

    def fake_repair(**kwargs):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(
        orchestrator,
        "generate_triton_kernel",
        fake_generate,
    )
    monkeypatch.setattr(orchestrator, "verify_candidate", fake_verify)
    monkeypatch.setattr(orchestrator, "repair_triton_kernel", fake_repair)

    with pytest.raises(RuntimeError):
        orchestrator.generate_verified_candidate(
            pytorch_code="reference",
            gpu_specs={},
            problem_file="problem.py",
            candidate_id=1,
        )

    assert client.closed


def test_gpu_verification_is_serialized(monkeypatch, tmp_path):
    state_lock = threading.Lock()
    active_verifications = 0
    maximum_active_verifications = 0

    monkeypatch.chdir(tmp_path)
    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(orchestrator, "create_openai_client", FakeClient)
    monkeypatch.setattr(
        orchestrator,
        "generate_triton_kernel",
        lambda pytorch_code, gpu_specs, client: "class ModelNew: pass",
    )

    def fake_verify_candidate(filename, problem_file):
        nonlocal active_verifications, maximum_active_verifications

        with state_lock:
            active_verifications += 1
            maximum_active_verifications = max(
                maximum_active_verifications,
                active_verifications,
            )

        time.sleep(0.02)

        with state_lock:
            active_verifications -= 1

        return {"passed": True, "tests": 1}

    monkeypatch.setattr(
        orchestrator,
        "verify_candidate",
        fake_verify_candidate,
    )

    candidates = orchestrator.generate_seed_candidates(
        pytorch_code="reference",
        gpu_specs={},
        problem_file="problem.py",
    )

    assert maximum_active_verifications == 1
    assert [candidate["id"] for candidate in candidates] == [1, 2, 3, 4]


def test_benchmarking_starts_after_seed_generation(monkeypatch, tmp_path):
    completed_seed_count = 0
    state_lock = threading.Lock()
    benchmark_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "get_gpu_specs", lambda: {})

    def fake_generate_verified_candidate(**kwargs):
        nonlocal completed_seed_count
        candidate_id = kwargs["candidate_id"]

        with state_lock:
            completed_seed_count += 1

        return {
            "id": candidate_id,
            "code": "class ModelNew: pass",
            "file": f"generated_kernel_seed_{candidate_id}.py",
        }

    def fake_benchmark_candidate(filename, problem_file):
        assert completed_seed_count == 4
        benchmark_calls.append(filename)
        return {
            "pytorch_ms": 2.0,
            "triton_ms": 1.0,
            "speedup": 2.0,
        }

    monkeypatch.setattr(
        orchestrator,
        "generate_verified_candidate",
        fake_generate_verified_candidate,
    )
    monkeypatch.setattr(
        orchestrator,
        "benchmark_candidate",
        fake_benchmark_candidate,
    )
    monkeypatch.setattr(
        orchestrator,
        "create_candidate_workload",
        lambda problem_file: None,
    )
    monkeypatch.setattr(orchestrator, "profile_candidate", lambda: None)

    orchestrator.run_optimization("reference", problem_file=None)

    assert benchmark_calls == [
        "generated_kernel_seed_1.py",
        "generated_kernel_seed_2.py",
        "generated_kernel_seed_3.py",
        "generated_kernel_seed_4.py",
    ]
