import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpu_agent.optimization import benchmark, benchmark_worker, orchestrator


@pytest.fixture
def cuda_clock(monkeypatch):
    def install(elapsed):
        values = iter(elapsed)
        events = []

        def event(**kwargs):
            assert kwargs == {"enable_timing": True}
            obj = Mock()
            obj.elapsed_time.side_effect = lambda end: next(values)
            events.append(obj)
            return obj

        sync = Mock()
        monkeypatch.setattr(benchmark.torch.cuda, "Event", event)
        monkeypatch.setattr(benchmark.torch.cuda, "synchronize", sync)
        return events, sync

    return install


def test_statistics():
    assert benchmark.timing_statistics([1, 2, 9]) == {
        "median_ms": 2, "mean_ms": 4, "stddev_ms": pytest.approx(19 ** 0.5),
        "min_ms": 1, "max_ms": 9, "num_samples": 3, "samples_ms": [1, 2, 9],
    }
    assert benchmark.timing_statistics([7])["stddev_ms"] == 0


@pytest.mark.parametrize("warmups,samples", [(10, 30), (0, 1), (3, 4)])
def test_repeated_sampling(cuda_clock, warmups, samples):
    events, sync = cuda_clock([40] * samples + [20] * samples)
    pytorch, triton = Mock(), Mock()
    kwargs = {} if (warmups, samples) == (10, 30) else {
        "warmup_count": warmups, "sample_count": samples,
    }
    result = benchmark.benchmark_functions(pytorch, triton, **kwargs)
    assert pytorch.call_count == triton.call_count == warmups + samples * 20
    assert len(events) == 4
    assert all(event.record.call_count == samples for event in events)
    assert sync.call_count == 1 + 2 * samples
    assert result["pytorch_stats"]["samples_ms"] == [2] * samples
    assert result["triton_stats"]["samples_ms"] == [1] * samples
    assert result["pytorch_ms"] == 2
    assert result["triton_ms"] == 1
    assert result["speedup"] == 2


@pytest.mark.parametrize("kwargs", [{"sample_count": 0}, {"warmup_count": -1},
                                      {"sample_count": 1.5}, {"warmup_count": True}])
def test_invalid_counts(kwargs):
    with pytest.raises(ValueError):
        benchmark.benchmark_functions(Mock(), Mock(), **kwargs)


def test_legacy_relu_path(monkeypatch, cuda_clock, capsys):
    cuda_clock([40, 60, 800, 20, 40, 600])
    reference, candidate = Mock(), Mock()
    monkeypatch.setattr(benchmark.torch, "randn", lambda *a, **k: "input")
    monkeypatch.setattr(benchmark.torch, "relu", reference)
    monkeypatch.setattr(benchmark, "load_module", lambda *a: SimpleNamespace(
        triton_implementation=candidate))
    result = benchmark.benchmark_candidate(warmup_count=2, sample_count=3)
    assert reference.call_count == candidate.call_count == 62
    assert result["pytorch_ms"] == 3
    assert result["triton_ms"] == 2
    assert result["speedup"] == 1.5
    output = capsys.readouterr().out
    assert "PyTorch: median" in output and "Triton: median" in output
    assert "stddev" in output and "3 samples" in output


@pytest.mark.parametrize("counts", [[], ["2", "3"]])
def test_kernelbench_worker(monkeypatch, cuda_clock, capsys, counts):
    warmups, samples = (2, 3) if counts else (10, 30)
    cuda_clock([40] * samples + [20] * samples)
    reference, candidate = Mock(), Mock()
    for model in (reference, candidate):
        model.cuda.return_value = model
    problem = SimpleNamespace(Model=Mock(return_value=reference),
                              get_init_inputs=lambda: [], get_inputs=lambda: [123])
    generated = SimpleNamespace(ModelNew=Mock(return_value=candidate))
    monkeypatch.setattr(benchmark_worker, "load_module",
                        lambda filename, name: generated if filename == "candidate.py" else problem)
    monkeypatch.setattr(benchmark_worker, "set_seed", lambda seed: None)
    monkeypatch.setattr(benchmark_worker.sys, "argv",
                        ["worker", "candidate.py", "problem.py", *counts])
    benchmark_worker.main()
    result = json.loads(capsys.readouterr().out)
    assert reference.call_count == candidate.call_count == warmups + samples * 20
    assert result["pytorch_stats"]["num_samples"] == samples
    assert result["triton_stats"]["samples_ms"] == [1] * samples
    assert result["speedup"] == 2


def test_subprocess_configuration(monkeypatch, capsys):
    result = {"pytorch_ms": 2, "triton_ms": 1, "speedup": 2,
              "pytorch_stats": benchmark.timing_statistics([2]),
              "triton_stats": benchmark.timing_statistics([1])}
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result)))
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.benchmark_candidate("candidate.py", "problem.py", 4, 7) == result
    assert run.call_args.args[0][-4:] == ["candidate.py", "problem.py", "4", "7"]
    assert "stddev" in capsys.readouterr().out


def test_median_selection_and_optimization(monkeypatch, tmp_path, cuda_clock):
    monkeypatch.chdir(tmp_path)
    # Seed 1 wins by median (2 vs 3), but loses by mean (34 vs 3).
    # V2 loses by median (3), despite a better mean; V3 wins by median (1).
    samples = iter([[2, 2, 98], [3, 3, 3], [3, 3, 3], [1, 1, 200]])

    def measured(*args, **kwargs):
        values = next(samples)
        cuda_clock([200] * 3 + [v * 20 for v in values])
        return benchmark.benchmark_functions(lambda: None, lambda: None,
                                             warmup_count=0, sample_count=3)

    monkeypatch.setattr(orchestrator, "get_gpu_specs", lambda: {})
    monkeypatch.setattr(orchestrator, "generate_seed_candidates", lambda **k: [
        {"id": i, "code": f"seed{i}", "file": f"seed{i}.py"} for i in (1, 2)])
    monkeypatch.setattr(orchestrator, "benchmark_candidate", measured)
    monkeypatch.setattr(orchestrator, "create_candidate_workload", lambda **k: None)
    profiles = iter([{"metrics": {}}, {"metrics": {}}, None])
    monkeypatch.setattr(orchestrator, "profile_candidate", lambda: next(profiles))
    monkeypatch.setattr(orchestrator, "analyze_roofline", lambda m: dict(
        classification="test", compute_sol=0, memory_sol=0, efficiency=0, headroom=0))
    optimized_from = []

    def optimize(**kwargs):
        optimized_from.append(kwargs["triton_code"])
        return f"candidate{len(optimized_from)}"

    monkeypatch.setattr(orchestrator, "optimize_triton_kernel", optimize)
    monkeypatch.setattr(orchestrator, "extract_python_code", lambda code: code)
    monkeypatch.setattr(orchestrator, "verify_candidate", lambda *a, **k: {
        "passed": True, "tests": 1})
    result = orchestrator.run_optimization("reference")
    assert result["seed"] == 1
    assert optimized_from == ["seed1", "seed1"]  # V2 was rejected.
    assert result["winner"] == "v3"
    assert result["benchmark"]["triton_ms"] == 1
    assert (tmp_path / "generated_kernel_best.py").read_text() == "candidate2"
