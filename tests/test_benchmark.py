import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpu_agent.optimization import benchmark, benchmark_worker, orchestrator


@pytest.fixture
def adaptive_timer(monkeypatch):
    def install(samples):
        values = iter(samples)

        def measure(call, **kwargs):
            assert not benchmark.torch.is_grad_enabled()
            call()
            return next(values)

        timer = Mock(side_effect=measure)
        monkeypatch.setattr(benchmark, "do_bench", timer)
        return timer

    return install


def test_statistics():
    assert benchmark.timing_statistics([1, 2, 9]) == {
        "median_ms": 2, "mean_ms": 4, "stddev_ms": pytest.approx(19 ** 0.5),
        "min_ms": 1, "max_ms": 9, "num_samples": 3, "samples_ms": [1, 2, 9],
    }
    assert benchmark.timing_statistics([7])["stddev_ms"] == 0


@pytest.mark.parametrize("warmup_ms,rep_ms", [(25, 100), (0, 1), (2.5, 7.5)])
@pytest.mark.parametrize("samples", [([2], [1]), ([2, 4], [1, 3, 9])])
def test_adaptive_sampling(adaptive_timer, capsys, warmup_ms, rep_ms, samples):
    timer = adaptive_timer(samples)
    pytorch, triton = Mock(), Mock()
    kwargs = {} if (warmup_ms, rep_ms) == (25, 100) else {
        "warmup_ms": warmup_ms, "rep_ms": rep_ms,
    }
    result = benchmark.benchmark_functions(pytorch, triton, **kwargs)
    # Only the timer invokes the callables: no extra fixed warmups/batches.
    assert pytorch.call_count == triton.call_count == 1
    assert timer.call_count == 2
    for invocation, call in zip(timer.call_args_list, (pytorch, triton)):
        assert invocation.args == (call,)
        assert invocation.kwargs == dict(warmup=warmup_ms, rep=rep_ms, return_mode="all")
    for name, values in zip(("pytorch", "triton"), samples):
        assert result[f"{name}_stats"] == benchmark.timing_statistics(values)
        assert result[f"{name}_ms"] == result[f"{name}_stats"]["median_ms"]
    assert result["speedup"] == result["pytorch_ms"] / result["triton_ms"]
    output = capsys.readouterr()
    assert output.out == ""
    assert "PyTorch timing" in output.err and "Triton timing" in output.err
    assert "Timing complete" in output.err


@pytest.mark.parametrize("name", ["warmup_ms", "rep_ms", "timeout_s"])
@pytest.mark.parametrize("value", [-1, True, "25", float("nan"), float("inf"), None])
def test_invalid_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        benchmark.benchmark_candidate(**{name: value})


@pytest.mark.parametrize("name", ["rep_ms", "timeout_s"])
def test_zero_positive_budget(name):
    with pytest.raises(ValueError, match=name):
        benchmark.benchmark_candidate(**{name: 0})


def test_legacy_relu_path(monkeypatch, adaptive_timer, capsys):
    adaptive_timer([[2, 3, 40], [1, 2, 30]])
    reference, candidate = Mock(), Mock()
    monkeypatch.setattr(benchmark.torch, "randn", lambda *a, **k: "input")
    monkeypatch.setattr(benchmark.torch, "relu", reference)
    monkeypatch.setattr(benchmark, "load_module", lambda *a: SimpleNamespace(
        triton_implementation=candidate))
    result = benchmark.benchmark_candidate(warmup_ms=2, rep_ms=3)
    assert reference.call_count == candidate.call_count == 1
    assert result["pytorch_ms"] == 3
    assert result["triton_ms"] == 2
    assert result["speedup"] == 1.5
    output = capsys.readouterr().out
    assert "PyTorch: median" in output and "Triton: median" in output
    assert "stddev" in output and "3 samples" in output


@pytest.mark.parametrize("budgets", [[], ["2.5", "3.5"]])
def test_kernelbench_worker(monkeypatch, adaptive_timer, capsys, budgets):
    warmup_ms, rep_ms = (2.5, 3.5) if budgets else (25, 100)
    timer = adaptive_timer([[2], [1, 1]])
    reference, candidate = Mock(), Mock()
    for model in (reference, candidate):
        model.cuda.return_value = model
    problem = SimpleNamespace(Model=Mock(return_value=reference),
                              get_init_inputs=lambda: [], get_inputs=lambda: [123])
    generated = SimpleNamespace(ModelNew=Mock(return_value=candidate))
    def noisy_load(filename, name):
        print("module setup output")
        return generated if filename == "candidate.py" else problem

    monkeypatch.setattr(benchmark_worker, "load_module", noisy_load)
    monkeypatch.setattr(benchmark_worker, "set_seed", lambda seed: None)
    monkeypatch.setattr(benchmark_worker.sys, "argv",
                        ["worker", "candidate.py", "problem.py", *budgets])
    benchmark_worker.main()
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert "module setup output" in output.err
    assert "Setup:" in output.err and "Timing complete" in output.err
    assert reference.call_count == candidate.call_count == 1
    assert result["pytorch_stats"]["num_samples"] == 1
    assert result["triton_stats"]["samples_ms"] == [1, 1]
    assert all(c.kwargs == dict(warmup=warmup_ms, rep=rep_ms, return_mode="all")
               for c in timer.call_args_list)
    assert result["speedup"] == 2


def test_subprocess_configuration(monkeypatch, capsys):
    result = {"pytorch_ms": 2, "triton_ms": 1, "speedup": 2,
              "pytorch_stats": benchmark.timing_statistics([2]),
              "triton_stats": benchmark.timing_statistics([1])}
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result)))
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.benchmark_candidate("candidate.py", "problem.py", 4, 7, timeout_s=12.5) == result
    assert run.call_args.args[0][-4:] == ["candidate.py", "problem.py", "4", "7"]
    assert run.call_args.args[0][0] == benchmark.sys.executable
    assert run.call_args.kwargs == dict(stdout=benchmark.subprocess.PIPE,
                                       stderr=None, text=True, timeout=12.5)
    assert "stddev" in capsys.readouterr().out


def test_subprocess_timeout(monkeypatch):
    run = Mock(side_effect=benchmark.subprocess.TimeoutExpired("worker", 0.1))
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out after 0.1 seconds for candidate.py"):
        benchmark.benchmark_candidate("candidate.py", "problem.py", timeout_s=0.1)


@pytest.mark.parametrize("timeout", [False, True])
def test_real_subprocess_channels_and_timeout(monkeypatch, capfd, timeout):
    # Exercise real pipe inheritance and timeout cleanup with a CPU-only child.
    real_run = benchmark.subprocess.run
    result = {"pytorch_ms": 2, "triton_ms": 1, "speedup": 2,
              "pytorch_stats": benchmark.timing_statistics([2]),
              "triton_stats": benchmark.timing_statistics([1])}
    script = (
        "import sys, time; print('child phase progress', file=sys.stderr, flush=True); "
        + ("time.sleep(30)" if timeout else f"print({json.dumps(result)!r})")
    )

    def run_child(command, **kwargs):
        return real_run([benchmark.sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "run", run_child)
    if timeout:
        with pytest.raises(RuntimeError, match="timed out"):
            benchmark.benchmark_candidate("candidate.py", "problem.py", timeout_s=0.5)
    else:
        assert benchmark.benchmark_candidate("candidate.py", "problem.py") == result
    output = capfd.readouterr()
    assert "child phase progress" in output.err
    assert "child phase progress" not in output.out


@pytest.mark.parametrize("returncode,stdout,message", [
    (1, '{"error": "failure"}', "Benchmark worker failed"),
    (0, 'bad json', "Could not parse benchmark worker output"),
])
def test_subprocess_errors(monkeypatch, returncode, stdout, message):
    monkeypatch.setattr(benchmark.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=returncode, stdout=stdout)))
    with pytest.raises(RuntimeError, match=message):
        benchmark.benchmark_candidate("candidate.py", "problem.py")


def test_worker_error_json(monkeypatch, capsys):
    monkeypatch.setattr(benchmark_worker.sys, "argv", ["worker", "c.py", "p.py", "-1"])
    with pytest.raises(SystemExit) as error:
        benchmark_worker.main()
    assert error.value.code == 1
    assert "warmup_ms" in json.loads(capsys.readouterr().out)["error"]


def test_median_selection_and_optimization(monkeypatch, tmp_path, adaptive_timer, capsys):
    monkeypatch.chdir(tmp_path)
    # Seed 1 wins by median (2 vs 3), but loses by mean (34 vs 3).
    # V2 loses by median (3), despite a better mean; V3 wins by median (1).
    samples = iter([[2, 2, 98], [3, 3, 3], [3, 3, 3], [1, 1, 200]])

    benchmark_starts = iter([
        "Benchmarking Seed 1...", "Benchmarking Seed 2...",
        "Benchmarking optimization candidate V2...", "Benchmarking optimization candidate V3...",
    ])

    def measured(*args, **kwargs):
        assert next(benchmark_starts) in capsys.readouterr().out
        values = next(samples)
        adaptive_timer([[10] * 3, values])
        return benchmark.benchmark_functions(lambda: None, lambda: None,
                                             warmup_ms=0, rep_ms=3)

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
