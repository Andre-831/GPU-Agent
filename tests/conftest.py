import os

import pytest


os.environ["OPENAI_API_KEY"] = "test-api-key"


@pytest.fixture(autouse=True)
def block_openai_api_requests(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("Tests must not make OpenAI API requests")

    monkeypatch.setattr("openai.OpenAI.request", fail_request)
    monkeypatch.setattr("openai.AsyncOpenAI.request", fail_request)
