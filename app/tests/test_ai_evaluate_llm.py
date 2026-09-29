"""Tests for ai_evaluate.evaluate_one's OpenAI-format call/response handling
— no real network calls, the client is a MagicMock. Verifies: (1) a clean
tool-call response is parsed into the evaluation dict, (2) invalid JSON in
the tool-call arguments is retried once and then raises, (3) invalid JSON
that succeeds on retry recovers, (4) an auth/quota/rate-limit error raises
FatalLLMError instead of being retried like an ordinary bad response.
Run with: python -m pytest app/tests/test_ai_evaluate_llm.py
"""
import json

import openai
import pytest
from unittest.mock import MagicMock

from app import ai_evaluate

PROFILE = {"identity": {"name": "Test Candidate"}, "competencies": {}}
JOB = {
    "company": "TestCo",
    "title": "Backend Engineer",
    "location": "Remote",
    "url": "https://testco.example/jobs/1",
    "description": "Build things with Python.",
}

EVALUATION = {
    "match_score": 80,
    "recommendation": "apply",
    "genuine_gaps": "None notable.",
    "transferable_strengths": "Directly relevant backend experience.",
    "risk_factors": "None notable.",
}


def _completion(arguments: str, finish_reason: str = "stop"):
    call = MagicMock()
    call.function.name = "submit_evaluation"
    call.function.arguments = arguments
    message = MagicMock(tool_calls=[call])
    choice = MagicMock(message=message, finish_reason=finish_reason)
    return MagicMock(choices=[choice])


def test_success_parses_tool_call():
    client = MagicMock()
    client.chat.completions.create.return_value = _completion(json.dumps(EVALUATION))

    result = ai_evaluate.evaluate_one(client, PROFILE, JOB)

    assert result == EVALUATION
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "submit_evaluation"
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit_evaluation"}}
    assert kwargs["messages"][0] == {"role": "system", "content": ai_evaluate.SYSTEM_PROMPT}


def test_invalid_json_retries_then_raises():
    client = MagicMock()
    client.chat.completions.create.return_value = _completion("{not valid json")

    with pytest.raises(RuntimeError, match="wasn't valid JSON"):
        ai_evaluate.evaluate_one(client, PROFILE, JOB)

    assert client.chat.completions.create.call_count == 2  # initial + 1 retry


def test_invalid_json_then_valid_on_retry_recovers():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion("{not valid json", finish_reason="length"),
        _completion(json.dumps(EVALUATION)),
    ]

    result = ai_evaluate.evaluate_one(client, PROFILE, JOB)

    assert result == EVALUATION
    assert client.chat.completions.create.call_count == 2
    second_call_max_tokens = client.chat.completions.create.call_args_list[1].kwargs["max_tokens"]
    first_call_max_tokens = client.chat.completions.create.call_args_list[0].kwargs["max_tokens"]
    assert second_call_max_tokens > first_call_max_tokens


@pytest.mark.parametrize("error", [
    openai.RateLimitError("quota exceeded", response=MagicMock(status_code=429), body=None),
    openai.AuthenticationError("invalid api key", response=MagicMock(status_code=401), body=None),
    openai.PermissionDeniedError("forbidden", response=MagicMock(status_code=403), body=None),
])
def test_auth_and_rate_limit_errors_abort_instead_of_retrying(error):
    client = MagicMock()
    client.chat.completions.create.side_effect = error

    with pytest.raises(ai_evaluate.FatalLLMError):
        ai_evaluate.evaluate_one(client, PROFILE, JOB)

    assert client.chat.completions.create.call_count == 1  # no retry on a fatal error
