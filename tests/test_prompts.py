from types import SimpleNamespace

import pytest

from gpu_agent.generation import generator
from gpu_agent.optimization import optimizer, orchestrator


@pytest.fixture
def model_client():
    prompts = []

    def create(**kwargs):
        prompts.append(kwargs['input'])
        return SimpleNamespace(output_text='class ModelNew: pass')

    return SimpleNamespace(responses=SimpleNamespace(create=create), prompts=prompts)


def failure(code):
    return dict(round=1, kernel_code=code, error_type='correctness',
                error='Small numerical mismatch')


@pytest.mark.parametrize('stage', ['generation', 'repair', 'optimization'])
def test_rendered_precision_guidance(stage, model_client, monkeypatch):
    code = '#' * 600 + '\nacc = tl.dot(a, b, input_precision="tf32")'
    history = [failure(code)]
    if stage == 'generation':
        generator.generate_triton_kernel('reference', {}, client=model_client)
    elif stage == 'repair':
        generator.repair_triton_kernel('reference', code, {}, 'correctness',
                                      'Small numerical mismatch', history,
                                      client=model_client)
    else:
        monkeypatch.setattr(optimizer, 'client', model_client)
        optimizer.optimize_triton_kernel(
            'reference', 'verified ieee kernel', {},
            dict(pytorch_ms=2, triton_ms=1, speedup=2), {},
            dict(classification='compute', compute_sol=50, memory_sol=10,
                 efficiency=50, headroom=50), refinement_history=history)
    prompt = model_client.prompts[0]
    assert 'Do not assume TF32 is acceptable' in prompt
    assert 'may fail strict correctness tolerances' in prompt
    assert 'input_precision="ieee"' in prompt
    assert 'Use verification results to decide' in prompt
    assert 'No candidate can be accepted unless it passes verification' in prompt
    if stage != 'generation':
        assert 'precision as a likely cause before' in prompt
        assert 'acc = tl.dot(a, b, input_precision="tf32")' in prompt
        assert 'Small numerical mismatch' in prompt
    if stage == 'optimization':
        assert 'Do not propose TF32 merely for speed' in prompt
        assert 'previous verification history shows' in prompt


def test_optimization_retains_failures_and_rejects_unverified_candidates(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seed_history = [failure('seed tl.dot(a, b, input_precision="tf32")')]
    seed = dict(id=1, code='verified ieee kernel', file='seed.py',
                refinement_history=seed_history)
    monkeypatch.setattr(orchestrator, 'get_gpu_specs', lambda: {})
    monkeypatch.setattr(orchestrator, 'generate_seed_candidates', lambda **kw: [seed])
    benchmarks = []

    def benchmark(filename, **kwargs):
        benchmarks.append(filename)
        return dict(pytorch_ms=2, triton_ms=1, speedup=2)

    monkeypatch.setattr(orchestrator, 'benchmark_candidate', benchmark)
    monkeypatch.setattr(orchestrator, 'create_candidate_workload', lambda **kw: None)
    monkeypatch.setattr(orchestrator, 'profile_candidate', lambda: {'metrics': {}})
    monkeypatch.setattr(orchestrator, 'analyze_roofline', lambda _: dict(
        classification='compute', compute_sol=50, memory_sol=10, efficiency=50, headroom=50))
    calls = []

    def optimize(**kwargs):
        calls.append(list(kwargs['refinement_history']))
        return f'failed dot candidate {len(calls)}'

    monkeypatch.setattr(orchestrator, 'optimize_triton_kernel', optimize)
    monkeypatch.setattr(orchestrator, 'verify_candidate', lambda *a, **kw: dict(
        passed=False, error_type='correctness', error='Small numerical mismatch'))
    result = orchestrator.run_optimization('reference')
    assert calls[0] == seed_history
    assert calls[1][-1]['kernel_code'] == 'failed dot candidate 1'
    assert calls[2][-1]['kernel_code'] == 'failed dot candidate 2'
    assert len(calls[-1]) == 10
    assert benchmarks == ['seed.py']
    assert result['code'] == 'verified ieee kernel'
    assert (tmp_path / 'generated_kernel_best.py').read_text() == 'verified ieee kernel'
