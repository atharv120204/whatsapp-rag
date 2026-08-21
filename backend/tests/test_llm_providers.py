"""
Tests for the chat-provider abstraction.

The agent holds its conversation in OpenAI's message shape and each provider
translates at its own boundary. Gemini's translation is the risky one: it has
no "tool" role, tool results ride on a user turn as function_response parts,
and consecutive results must be merged into a single turn or the API rejects
the conversation. None of that is visible until a multi-step agent run fails
halfway, so it is pinned down here.

The OpenAI-compatible path is exercised against a stubbed transport, which
covers the response shapes that actually differ between hosts: absent content,
tool calls with string arguments, and malformed argument JSON.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.llm import (LLMError, OpenAICompatProvider,  # noqa: E402
                           PRESETS, openai_tool_schema)
from app.agent.tools import TOOL_DECLARATIONS  # noqa: E402


# --- tool schema ---------------------------------------------------------------

def test_tool_schema_wraps_every_declaration():
    schema = openai_tool_schema(TOOL_DECLARATIONS)
    assert len(schema) == len(TOOL_DECLARATIONS)
    for entry in schema:
        assert entry["type"] == "function"
        function = entry["function"]
        assert function["name"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"


def test_tool_with_no_parameters_still_gets_a_valid_schema():
    """get_overview takes nothing; an empty schema must still be well formed."""
    schema = {e["function"]["name"]: e["function"] for e
              in openai_tool_schema(TOOL_DECLARATIONS)}
    overview = schema["get_overview"]
    assert overview["parameters"]["type"] == "object"
    assert isinstance(overview["parameters"].get("properties"), dict)


def test_every_preset_is_complete():
    for pid, preset in PRESETS.items():
        assert preset["label"], pid
        assert "base_url" in preset, pid
        assert "default_model" in preset, pid
        if pid not in ("custom", "gemini"):
            assert preset["base_url"].startswith("http"), pid


# --- OpenAI-compatible transport ------------------------------------------------

class _StubResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


def _provider(monkeypatched_post):
    import httpx

    httpx.post = monkeypatched_post
    return OpenAICompatProvider(base_url="https://example.test/v1",
                                api_key="k", model="m", provider="groq")


def test_plain_text_answer_is_returned():
    import httpx

    original = httpx.post
    try:
        provider = _provider(lambda *a, **k: _StubResponse(
            {"choices": [{"message": {"content": "Rohit sent 1,239."}}]}))
        result = provider.complete([{"role": "user", "content": "hi"}], None)
        assert result.text == "Rohit sent 1,239."
        assert not result.wants_tools
    finally:
        httpx.post = original


def test_tool_calls_are_parsed_with_string_arguments():
    import httpx

    original = httpx.post
    try:
        provider = _provider(lambda *a, **k: _StubResponse({
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "run_sql",
                                 "arguments": '{"query": "SELECT 1"}'},
                }],
            }}]
        }))
        result = provider.complete([{"role": "user", "content": "hi"}], [])
        assert result.wants_tools
        call = result.tool_calls[0]
        assert call.name == "run_sql"
        assert call.arguments == {"query": "SELECT 1"}
        assert call.id == "call_1"
        assert result.text == ""          # null content must not crash
    finally:
        httpx.post = original


def test_malformed_tool_arguments_do_not_raise():
    """A model emitting broken JSON should fail the tool, not the whole turn."""
    import httpx

    original = httpx.post
    try:
        provider = _provider(lambda *a, **k: _StubResponse({
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "c", "function": {
                    "name": "run_sql", "arguments": "{not json"}}],
            }}]
        }))
        result = provider.complete([{"role": "user", "content": "hi"}], [])
        assert result.wants_tools
        assert "__malformed__" in result.tool_calls[0].arguments
    finally:
        httpx.post = original


def test_http_error_becomes_a_readable_llm_error():
    import httpx

    original = httpx.post
    try:
        provider = _provider(lambda *a, **k: _StubResponse(
            {}, status_code=401, text="invalid api key"))
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except LLMError as exc:
            assert "401" in str(exc) and "groq" in str(exc)
        else:
            raise AssertionError("expected LLMError")
    finally:
        httpx.post = original


def test_missing_base_url_is_rejected_early():
    try:
        OpenAICompatProvider(base_url="", api_key="k", model="m",
                             provider="custom")
    except LLMError as exc:
        assert "base URL" in str(exc)
    else:
        raise AssertionError("expected LLMError")


# --- Gemini message translation -------------------------------------------------

def _convert(messages):
    from google.genai import types

    from app.agent.llm import GeminiProvider

    return GeminiProvider._to_gemini(messages, types)


def test_system_messages_are_lifted_out():
    system, contents = _convert([
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hello"},
    ])
    assert system == "you are a bot"
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_assistant_maps_to_model_role():
    _, contents = _convert([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert [c.role for c in contents] == ["user", "model"]


def test_tool_calls_and_results_round_trip():
    _, contents = _convert([
        {"role": "user", "content": "how many?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "run_sql",
                                      "arguments": '{"query": "SELECT 1"}'}}]},
        {"role": "tool", "tool_call_id": "1", "name": "run_sql",
         "content": '{"row_count": 1}'},
    ])
    assert [c.role for c in contents] == ["user", "model", "user"]

    call_part = contents[1].parts[0]
    assert call_part.function_call.name == "run_sql"
    assert dict(call_part.function_call.args) == {"query": "SELECT 1"}

    result_part = contents[2].parts[0]
    assert result_part.function_response.name == "run_sql"


def test_parallel_tool_results_merge_into_one_turn():
    """
    Two results in a row must not become two user turns.

    Gemini rejects a conversation where a model turn is followed by more than
    one user turn of function responses, and the agent issues parallel calls
    whenever the model asks for them.
    """
    _, contents = _convert([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "1", "type": "function",
              "function": {"name": "run_sql", "arguments": "{}"}},
             {"id": "2", "type": "function",
              "function": {"name": "get_overview", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "1", "name": "run_sql",
         "content": "{}"},
        {"role": "tool", "tool_call_id": "2", "name": "get_overview",
         "content": "{}"},
    ])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert len(contents[1].parts) == 2       # both calls on one model turn
    assert len(contents[2].parts) == 2       # both results on one user turn


def test_non_json_tool_output_is_still_accepted():
    _, contents = _convert([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "run_sql", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "run_sql",
         "content": "not json at all"},
    ])
    assert contents[-1].parts[0].function_response.name == "run_sql"


def test_empty_assistant_content_is_dropped():
    """A blank turn carries nothing and some providers reject it."""
    _, contents = _convert([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "still there?"},
    ])
    assert [c.role for c in contents] == ["user", "user"]


# --- rate limiting --------------------------------------------------------------

GROQ_TPM_429 = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-20b` in '
    'organization `org_x` service tier `on_demand` on tokens per minute (TPM): '
    'Limit 8000, Used 5803, Requested 5313. Please try again in 23.37s. Need '
    'more tokens? Upgrade to Dev Tier today at '
    'https://console.groq.com/settings/billing","type":"tokens",'
    '"code":"rate_limit_exceeded"}}'
)


class _Headers(dict):
    def get(self, key, default=""):
        return dict.get(self, key.lower(), default)


class _Resp:
    def __init__(self, payload=None, status_code=200, text="", headers=None):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text or "{}"
        self.headers = _Headers(headers or {})

    def json(self):
        return self._payload


def test_wait_hint_is_read_from_the_real_groq_error():
    from app.agent.llm import _retry_delay

    delay = _retry_delay(_Resp(), GROQ_TPM_429, 0)
    # Just over the 23.37s the provider asked for, not a blind backoff.
    assert 23.0 < delay < 26.0, delay


def test_retry_after_header_wins_over_body_hint():
    from app.agent.llm import _retry_delay

    assert _retry_delay(_Resp(headers={"retry-after": "5"}), GROQ_TPM_429, 0) == 5.0


def test_delay_is_capped_so_a_bad_hint_cannot_hang():
    from app.agent.llm import _retry_delay

    assert _retry_delay(_Resp(), "try again in 100000s", 0) <= 90.0


def test_backoff_grows_without_a_hint():
    from app.agent.llm import _retry_delay

    first = _retry_delay(_Resp(), "internal error", 0)
    later = _retry_delay(_Resp(), "internal error", 3)
    assert later > first


def test_rate_limited_request_is_retried_and_succeeds():
    """A 429 carrying a wait hint must not end the turn."""
    import httpx

    from app.agent import llm

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(status_code=429, text=GROQ_TPM_429)
        return _Resp({"choices": [{"message": {"content": "done"}}]})

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = fake_post
        llm.time.sleep = lambda _s: None          # do not actually wait
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k", model="m",
            provider="groq")
        result = provider.complete([{"role": "user", "content": "hi"}], None)
        assert result.text == "done"
        assert calls["n"] == 2
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


def test_persistent_rate_limit_eventually_reports_clearly():
    import httpx

    from app.agent import llm

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = lambda *a, **k: _Resp(status_code=429, text=GROQ_TPM_429)
        llm.time.sleep = lambda _s: None
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k", model="m",
            provider="groq")
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except llm.LLMError as exc:
            assert "429" in str(exc) and "attempts" in str(exc)
        else:
            raise AssertionError("expected LLMError")
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


def test_client_errors_are_not_retried():
    """A bad key is not going to fix itself; fail fast instead of waiting."""
    import httpx

    from app.agent import llm

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _Resp(status_code=401, text="invalid api key")

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = fake_post
        llm.time.sleep = lambda _s: None
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="bad", model="m",
            provider="groq")
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except llm.LLMError:
            pass
        assert calls["n"] == 1, f"retried a 401 {calls['n']} times"
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


# --- context budget -------------------------------------------------------------

def test_short_conversation_is_left_alone():
    from app.agent.router import _trim_conversation

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "tool", "name": "run_sql", "content": '{"row_count": 1}'},
    ]
    assert _trim_conversation(messages) == messages


def test_old_tool_results_are_shrunk_but_recent_ones_kept():
    from app.agent.router import _trim_conversation
    from app.config import settings

    big = json.dumps({"row_count": 400, "rows": [["x" * 50] for _ in range(200)]})
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    for _ in range(6):
        messages.append({"role": "tool", "name": "run_sql", "content": big})

    trimmed = _trim_conversation(messages)
    total = sum(len(m.get("content") or "") for m in trimmed)

    assert total < sum(len(m.get("content") or "") for m in messages)
    # The most recent result is preserved -- truncated if it must be, but never
    # replaced by a stub, because it is what the model is reasoning about.
    assert "_trimmed" not in trimmed[-1]["content"]
    assert trimmed[-1]["content"].startswith(big[:200])
    # Something earlier was reduced, and kept its shape rather than vanishing.
    assert any("_trimmed" in (m.get("content") or "") for m in trimmed[:-1])
    assert total <= settings.context_budget_chars * 2 + 200


def test_system_prompt_is_never_trimmed():
    """The schema lives in the system prompt; losing it breaks SQL writing."""
    from app.agent.router import _trim_conversation

    system = "SCHEMA " + "s" * 20_000
    big = json.dumps({"row_count": 1, "rows": [["y" * 8000]]})
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "q"},
                {"role": "tool", "name": "run_sql", "content": big},
                {"role": "tool", "name": "run_sql", "content": big}]

    trimmed = _trim_conversation(messages)
    assert trimmed[0]["content"] == system


def test_summary_keeps_the_shape_of_what_was_dropped():
    from app.agent.router import _summarise_tool_result

    summary = _summarise_tool_result(json.dumps({
        "row_count": 108, "rows": [["a"] * 5] * 108,
        "sql": "SELECT date, COUNT(*) FROM v_messages"}))
    assert "108" in summary
    assert "SELECT" in summary
    assert "_trimmed" in summary


def test_unparseable_tool_output_still_shrinks():
    from app.agent.router import _summarise_tool_result

    summary = _summarise_tool_result("x" * 5000)
    assert len(summary) < 500



GROQ_413 = (
    '{"error":{"message":"Request too large for model `openai/gpt-oss-120b` in '
    'organization `org_x` service tier `on_demand` on tokens per minute (TPM): '
    'Limit 8000, Requested 8126, please reduce your message size and try '
    'again.","type":"tokens","code":"rate_limit_exceeded"}}'
)


def test_oversized_request_is_distinguished_from_rate_limiting():
    """Both arrive as a token error; only one is fixed by waiting."""
    from app.agent.llm import is_too_large

    assert is_too_large(GROQ_413)
    assert not is_too_large(GROQ_TPM_429)


def test_oversized_request_is_not_retried():
    import httpx

    from app.agent import llm

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _Resp(status_code=413, text=GROQ_413)

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = fake_post
        llm.time.sleep = lambda _s: None
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k", model="m",
            provider="groq")
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except llm.LLMError:
            pass
        assert calls["n"] == 1, f"retried an oversized request {calls['n']} times"
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


def test_agent_retries_smaller_when_refused_as_too_large():
    """
    A refusal for size must shrink and retry, not end the turn.

    The observed conversation size can plateau once every old result is already
    a stub, so what is asserted is the behaviour that matters: it keeps trying
    with a smaller budget and eventually succeeds.
    """
    from app.agent import router
    from app.agent.llm import LLMError

    attempts = {"n": 0}

    class FakeProvider:
        name = "fake"

        def complete(self, messages, tools, temperature=0.2):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise LLMError("413: Request too large, please reduce your "
                               "message size and try again")
            return type("R", (), {"text": "ok", "tool_calls": [],
                                  "wants_tools": False})()

    big = json.dumps({"row_count": 1, "rows": [["z" * 400]] * 40})
    messages = [{"role": "system", "content": "sys"}]
    for _ in range(8):
        messages.append({"role": "tool", "name": "run_sql", "content": big})

    result = router._complete(FakeProvider(), messages, None)
    assert result.text == "ok"
    assert attempts["n"] == 3


def test_a_genuinely_unfixable_size_error_still_surfaces():
    """Shrinking has a floor; after that the user must be told."""
    from app.agent import router
    from app.agent.llm import LLMError

    class AlwaysTooBig:
        name = "fake"

        def complete(self, messages, tools, temperature=0.2):
            raise LLMError("413: Request too large, reduce your message size")

    try:
        router._complete(AlwaysTooBig(), [{"role": "user", "content": "q"}], None)
    except LLMError as exc:
        assert "too large" in str(exc).lower()
    else:
        raise AssertionError("expected LLMError")


def test_budget_excludes_the_system_prompt():
    """
    Otherwise the setting means something different at every archive size.

    The system prompt grows with the participant list, so counting it inside
    the budget would silently leave less and less room for tool results.
    """
    from app.agent.router import _message_size, _trim_conversation

    system = {"role": "system", "content": "S" * 20_000}
    small = {"role": "tool", "name": "run_sql", "content": '{"row_count": 1}'}
    trimmed = _trim_conversation([system, small], budget=8_000)

    # Well over the budget in total, but the conversation itself fits, so
    # nothing is touched.
    assert trimmed[1]["content"] == small["content"]
    assert len(trimmed[0]["content"]) == 20_000
    assert sum(_message_size(m) for m in trimmed if m["role"] != "system") < 8_000


def test_message_size_counts_tool_call_arguments():
    from app.agent.router import _message_size

    assert _message_size({"role": "user", "content": "hello"}) == 5
    with_calls = {"role": "assistant", "content": "",
                  "tool_calls": [{"function": {"name": "run_sql",
                                               "arguments": "x" * 300}}]}
    assert _message_size(with_calls) > 300



GROQ_TPD = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    'organization `org_x` service tier `on_demand` on tokens per day (TPD): '
    'Limit 200000, Used 198074, Requested 3453. Please try again in '
    '10m59.664s.","type":"tokens","code":"rate_limit_exceeded"}}'
)


def test_daily_limit_is_distinguished_from_per_minute():
    from app.agent.llm import is_daily_limit

    assert is_daily_limit(GROQ_TPD)
    assert not is_daily_limit(GROQ_TPM_429)


def test_daily_limit_is_not_retried():
    """
    Waiting cannot fix a quota that resets tomorrow.

    This cost four and a half minutes of dead waiting: the provider asked for
    eleven minutes, the delay was capped at ninety seconds, and it retried
    three times before failing anyway.
    """
    import httpx

    from app.agent import llm

    calls = {"n": 0}
    slept = []

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _Resp(status_code=429, text=GROQ_TPD)

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = fake_post
        llm.time.sleep = lambda s: slept.append(s)
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k",
            model="openai/gpt-oss-120b", provider="groq")
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except llm.LLMError as exc:
            assert "no quota left today" in str(exc)
            assert "Settings tab" in str(exc)
        else:
            raise AssertionError("expected LLMError")

        assert calls["n"] == 1, f"retried a daily limit {calls['n']} times"
        assert not slept, f"slept {slept} on a daily limit"
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


def test_a_wait_longer_than_anyone_will_sit_through_fails_fast():
    """A minute of spinner is tolerable; ten is not."""
    import httpx

    from app.agent import llm

    slept = []
    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = lambda *a, **k: _Resp(
            status_code=429, text="Rate limited, please try again in 400s")
        llm.time.sleep = lambda s: slept.append(s)
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k", model="m",
            provider="groq")
        try:
            provider.complete([{"role": "user", "content": "hi"}], None)
        except llm.LLMError as exc:
            assert "rate limited" in str(exc).lower()
        assert not slept, f"slept {slept} instead of failing fast"
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep


def test_short_waits_are_still_retried():
    """The fast-fail must not swallow ordinary, brief throttling."""
    import httpx

    from app.agent import llm

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(status_code=429,
                         text="Rate limited, please try again in 3s")
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    original_post, original_sleep = httpx.post, llm.time.sleep
    try:
        httpx.post = fake_post
        llm.time.sleep = lambda _s: None
        provider = llm.OpenAICompatProvider(
            base_url="https://example.test/v1", api_key="k", model="m",
            provider="groq")
        assert provider.complete([{"role": "user", "content": "hi"}], None).text == "ok"
        assert calls["n"] == 2
    finally:
        httpx.post, llm.time.sleep = original_post, original_sleep



# --- Gemini's limiter reads its own 429s ----------------------------------------
#
# Real bodies, kept verbatim. The embedding metric is the trap: it is named
# `embed_content_free_tier_requests` but capped per *minute*, so a substring
# test for "free_tier_requests" reads it as the day being spent and abandons
# the run after one batch. That halted an embedding backfill at 64 of 541
# chunks; the wait it actually needed was 16 seconds.

GEMINI_EMBED_PER_MINUTE_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota, please check your plan and billing details.\\n"
    "* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/embed_content_free_tier_requests, "
    "limit: 100, model: gemini-embedding-1.0\\nPlease retry in 15.971685376s.', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier', "
    "'quotaValue': '100'}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '15s'}]}}"
)

GEMINI_FLASH_PER_DAY_429 = (
    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-2.5-flash. quotaId: "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', quotaValue: '20'"
)


def test_gemini_per_minute_quota_is_not_read_as_the_daily_one():
    from app.index.ratelimit import _is_daily_quota

    assert not _is_daily_quota(GEMINI_EMBED_PER_MINUTE_429)
    assert _is_daily_quota(GEMINI_FLASH_PER_DAY_429)


def test_gemini_states_its_own_wait_and_we_believe_it():
    from app.index.ratelimit import retry_after

    stated = retry_after(GEMINI_EMBED_PER_MINUTE_429)
    assert stated is not None and 15.0 < stated < 16.5, stated
    # A daily refusal states no wait, because waiting is not the answer.
    assert retry_after(GEMINI_FLASH_PER_DAY_429) is None


def test_a_per_minute_gemini_429_does_not_burn_the_days_budget():
    """
    The regression that mattered: note_throttled() adopting a per-minute limit
    as the daily cap, so the very next acquire() raised DailyQuotaReached.
    """
    from app.config import settings
    from app.index.ratelimit import RateLimiter

    limiter = RateLimiter()
    limiter._persist_enabled = False   # do not write to the real usage table
    limiter._budget(MODEL).loaded = True   # and do not read from it either
    original_rpm, original_rpd = (settings.max_requests_per_minute,
                                  settings.max_requests_per_day)
    try:
        settings.max_requests_per_minute = 0     # no spacing, so the test is fast
        settings.max_requests_per_day = 20

        limiter.acquire(MODEL)
        limiter.note_throttled(GEMINI_EMBED_PER_MINUTE_429, MODEL)

        assert settings.max_requests_per_day == 20, "adopted a per-minute limit"
        assert limiter._budget(MODEL).stated_limit is None, "adopted it as a ceiling"
        assert limiter._budget(MODEL).count == 1, limiter._budget(MODEL).count
        limiter.acquire(MODEL)   # must not raise
    finally:
        settings.max_requests_per_minute = original_rpm
        settings.max_requests_per_day = original_rpd


# --- the daily budget is per model, and it persists -----------------------------
#
# Per model because one shared counter conflated limits two orders of magnitude
# apart: a vision 429 (~20/day) capped the embedding model (~1,000/day) for the
# rest of the day, semantic search went dark, and the agent burned eight Groq
# calls re-searching by keyword for a question that needed two.

MODEL = "test-model"
OTHER_MODEL = "other-model"


class _FakeUsageStore:
    """
    Stands in for the device's cache database.

    The limiter reaches for `get_cache_connection()` inside each call rather
    than holding a handle, so redirecting it at the module is enough -- and it
    keeps these tests from writing into the real usage table, which would
    silently spend the user's actual daily budget.
    """

    def __init__(self):
        import duckdb

        self.conn = duckdb.connect(":memory:")

    def rows(self, model=MODEL):
        try:
            return self.conn.execute(
                "SELECT day, requests FROM api_usage_by_key "
                "WHERE model = ? ORDER BY day", [model]).fetchall()
        except Exception:      # table not created yet
            return []


def _current_credential():
    """The fingerprint the limiter will be counting under right now."""
    from app.index.ratelimit import RateLimiter

    return RateLimiter.current_credential()


def _with_fake_usage(cap, body):
    """Run `body(store)` with usage persistence pointed at an in-memory table."""
    from app import db as app_db
    from app.config import settings

    store = _FakeUsageStore()
    original = app_db.get_cache_connection
    rpm, rpd = settings.max_requests_per_minute, settings.max_requests_per_day
    try:
        app_db.get_cache_connection = lambda: store.conn
        settings.max_requests_per_minute = 0     # no spacing, so tests are fast
        settings.max_requests_per_day = cap
        return body(store)
    finally:
        app_db.get_cache_connection = original
        settings.max_requests_per_minute, settings.max_requests_per_day = rpm, rpd


def test_the_first_request_of_the_day_is_written_out_immediately():
    """
    The counter used to persist only on multiples of 20, under a cap of 20 --
    so the row was first written by the request that had already exhausted the
    budget, and every earlier one was invisible to the next process.
    """
    from app.index.ratelimit import RateLimiter

    def body(store):
        limiter = RateLimiter()
        limiter.acquire(MODEL)
        assert store.rows(), "one request left no trace on disk"
        assert store.rows()[0][1] == 1, store.rows()

    _with_fake_usage(20, body)


def test_the_daily_cap_survives_a_restart():
    from app.index.ratelimit import DailyQuotaReached, RateLimiter

    def body(store):
        spent = RateLimiter()
        spent.acquire(MODEL)
        spent.acquire(MODEL)

        # A second process, as a later CLI invocation would be: nothing carried
        # over in memory, so the cap holds only if it was read back from disk.
        fresh = RateLimiter()
        fresh.acquire(MODEL)
        count = fresh._budget(MODEL).count
        assert count == 3, f"did not read back 2 spent requests: {count}"

        try:
            fresh.acquire(MODEL)
        except DailyQuotaReached:
            pass
        else:
            raise AssertionError("cap did not survive the restart")

    _with_fake_usage(3, body)


def test_yesterdays_spending_does_not_count_against_today():
    from datetime import date, timedelta

    from app.index.ratelimit import RateLimiter

    def body(store):
        limiter = RateLimiter()
        limiter.acquire(MODEL)                  # creates the table
        store.conn.execute(
            "INSERT INTO api_usage_by_key VALUES (?, ?, ?, ?) "
            "ON CONFLICT (day, model, credential) DO NOTHING",
            [date.today() - timedelta(days=1), MODEL,
             _current_credential(), 999],
        )

        fresh = RateLimiter()
        fresh.acquire(MODEL)
        assert fresh._budget(MODEL).count == 2, fresh._budget(MODEL).count

    _with_fake_usage(5, body)


def test_one_model_running_out_does_not_block_another():
    """
    The bug this whole split exists for. A vision 429 states limit: 20; the
    embedding model must keep its own budget, or semantic search dies for the
    day and every retrieval question falls back to keyword-only.
    """
    from app.index.ratelimit import DailyQuotaReached, RateLimiter

    def body(store):
        limiter = RateLimiter()
        limiter.acquire(MODEL)
        limiter.note_throttled(GEMINI_FLASH_PER_DAY_429, MODEL)   # limit: 20

        try:
            limiter.acquire(MODEL)
        except DailyQuotaReached:
            pass
        else:
            raise AssertionError("spent model kept going")

        # The other model has spent nothing and must be unaffected.
        for _ in range(5):
            limiter.acquire(OTHER_MODEL)
        assert limiter._budget(OTHER_MODEL).count == 5

    _with_fake_usage(200, body)


def test_step_budget_matches_the_shape_of_the_question():
    """
    A single budget is wrong in one direction or the other. At 4, "what did we
    decide about X" question ran out of steps and returned nothing at all; at 8, a
    one-query counting question was given room to wander.
    """
    from app.agent.router import step_budget
    from app.config import settings

    direct = settings.agent_max_steps_direct
    reasoning = settings.agent_max_steps_reasoning
    default = settings.agent_max_steps
    assert direct < default < reasoning, "budgets must be ordered"

    cases = [
        # The question that failed, and its family.
        ("what did we decide about the trip", reasoning),
        ("why did they stop replying", reasoning),
        ("did we ever argue about money", reasoning),
        ("summarise the trip planning", reasoning),
        # One computable answer.
        ("how many messages did each person send", direct),
        ("who starts conversations most often", direct),
        ("what time of day is the group most active", direct),
        # Neither marker: the middle budget.
        ("show me the photos of food", default),
    ]
    for question, expected in cases:
        got = step_budget(question)
        assert got == expected, f"{question!r}: {got} != {expected}"

    # Ambiguous questions must round up, because too few steps returns nothing
    # while too many is merely slow.
    assert step_budget("how many times did we argue") == reasoning


def test_reasoning_is_kept_out_of_the_answer_text():
    """
    Groq's gpt-oss returns its thinking separately. It must stay separate: on a
    normal turn it is private monologue, and merging it into `text` would print
    the model's thinking to the user.
    """
    import httpx

    original = httpx.post
    try:
        provider = _provider(lambda *a, **k: _StubResponse({
            "choices": [{"message": {
                "content": None,
                "reasoning": "The user asks about the escrow rule. I should search.",
            }}]
        }))
        result = provider.complete([{"role": "user", "content": "hi"}], None)
        assert result.text == "", "reasoning leaked into the answer"
        assert result.reasoning == "The user asks about the escrow rule. I should search."
    finally:
        httpx.post = original


def test_the_closing_call_may_fall_back_to_reasoning():
    """
    The one place it is legitimate: no tools were offered and the model was told
    to write the final answer, so a reply that landed in `reasoning` *is* the
    answer. Reading only `content` there showed "Stopped after N tool steps
    without reaching an answer" over a finished reply.
    """
    from app.agent import router
    from app.agent.llm import LLMResponse

    class _Provider:
        def complete(self, msgs, tools, **kwargs):
            return LLMResponse(text="", reasoning="Nothing was decided.")

    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "what did we decide?"},
        {"role": "tool", "tool_call_id": "1", "name": "search_chat",
         "content": '{"result_count": 0}'},
    ]
    assert router._force_answer(_Provider(), messages) == "Nothing was decided."


def test_the_closing_call_is_size_bounded():
    """
    Flattening the history without a cap sent 14,629 tokens at an 8,000/minute
    ceiling -- and because it is a single message, the shrink-and-retry could not
    reduce it, so the answer was lost at the very last step of an eight-step
    question. Newest results are kept: they are what the model was working on.
    """
    from app.agent import router
    from app.agent.llm import LLMResponse
    from app.config import settings

    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "explain the escrow rule"}]
    for i in range(8):
        messages.append({"role": "tool", "tool_call_id": str(i),
                         "name": "search_chat",
                         "content": ("X" * 3000) + f"#{i}"})

    seen = {}

    class _Provider:
        def complete(self, msgs, tools, **kwargs):
            seen["msgs"] = msgs
            return LLMResponse(text="ok")

    router._force_answer(_Provider(), messages)
    body = seen["msgs"][1]["content"]

    assert len(body) < settings.context_budget_chars + 800, \
        f"closing call unbounded at {len(body)} chars"
    assert "explain the escrow rule" in body, "dropped the question"
    assert "#7" in body, "dropped the newest result"
    assert "#0" not in body, "kept the oldest result over the newest"


def test_an_answer_with_no_tool_call_is_refused_once():
    """
    Asked "explain the escrow rule", the model wrote a confident essay about
    long-term procurement agreements -- from its own world knowledge, with no
    tool call, about a chat it cannot see. Preventing exactly that is the point
    of the app, so a tool-free first answer is pushed back on.
    """
    from app.agent import router
    from app.agent.llm import LLMResponse, ToolCallRequest

    replies = [
        # First: answers from general knowledge, no tools.
        LLMResponse(text="An escrow rule is a contractual arrangement..."),
        # After the nudge: searches instead.
        LLMResponse(tool_calls=[ToolCallRequest(
            id="1", name="search_chat", arguments={"query": "escrow rule"})]),
        LLMResponse(text="Nothing in the chat mentions an escrow rule."),
    ]
    sent: list[dict] = []

    class _Provider:
        name = "stub"

        def complete(self, msgs, tools, **kwargs):
            sent.append(msgs[-1])
            return replies.pop(0)

    def _prepare(question, history, archive):
        return (_Provider(), {"search_chat": lambda **kw: {"result_count": 0}},
                [], [{"role": "system", "content": "SYS"},
                     {"role": "user", "content": question}])

    original = router._prepare
    router._prepare = _prepare
    try:
        answer = router.ask("explain the escrow rule")
    finally:
        router._prepare = original

    assert "escrow rule is a contractual" not in answer.text, \
        "served an answer that never consulted the archive"
    assert answer.text == "Nothing in the chat mentions an escrow rule."
    assert [c.name for c in answer.tool_calls] == ["search_chat"]
    assert any("not looked at the archive" in (m.get("content") or "")
               for m in sent), "no push-back was sent"


def test_the_forced_answer_carries_no_tool_call_turns():
    """
    The last call of a turn must not be able to trigger Groq's
    `400 tool_use_failed: "Tool choice is none, but model called a tool"`.

    With assistant tool-call turns still in the transcript, gpt-oss emits
    another tool call and Groq rejects the whole request -- observed on a
    question whose first SQL call had already produced the answer: four steps of
    work, then an empty reply. Flattening to system + one user turn also drops
    the tool schema, making the final call the cheapest of the turn.
    """
    from app.agent import router

    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "who starts conversations?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "function": {"name": "run_sql",
                                                 "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "run_sql",
         "content": '{"rows": [["Rohit", 214]]}'},
    ]

    seen = {}

    class _Provider:
        def complete(self, msgs, tools, **kwargs):
            seen["msgs"], seen["tools"] = msgs, tools

            class _R:
                text = "Rohit, 214 times."
            return _R()

    assert router._force_answer(_Provider(), messages) == "Rohit, 214 times."
    assert [m["role"] for m in seen["msgs"]] == ["system", "user"]
    assert not any(m.get("tool_calls") for m in seen["msgs"])
    assert seen["tools"] is None, "tool schema still sent on the closing call"

    body = seen["msgs"][1]["content"]
    assert "who starts conversations?" in body, "lost the question"
    # The result that held the answer must survive the flattening.
    assert "Rohit" in body and "214" in body


def test_swapping_the_api_key_gives_you_that_key_s_own_budget():
    """
    Quotas belong to a key, not to a calendar day.

    Counting per (day, model) alone made "change the API key" useless: the
    limiter went on refusing requests because the *day* was spent, when what
    was spent was the previous key's allowance.
    """
    from app.config import settings
    from app.index.ratelimit import DailyQuotaReached, RateLimiter

    def body(store):
        original_key = settings.api_key
        try:
            settings.api_key = "AQ.first-key"
            limiter = RateLimiter()
            limiter.acquire(MODEL)
            limiter.note_throttled(GEMINI_FLASH_PER_DAY_429, MODEL)   # limit: 20

            try:
                limiter.acquire(MODEL)
            except DailyQuotaReached as exc:
                assert "Settings tab" in str(exc), "no way out offered"
            else:
                raise AssertionError("spent key kept going")

            # A different key: its own project, its own quota.
            settings.api_key = "AQ.second-key"
            for _ in range(5):
                limiter.acquire(MODEL)
            assert limiter._budget(MODEL).count == 5

            # And the spent one is still spent, not reset by the detour.
            settings.api_key = "AQ.first-key"
            try:
                limiter.acquire(MODEL)
            except DailyQuotaReached:
                pass
            else:
                raise AssertionError("the spent key was quietly forgiven")
        finally:
            settings.api_key = original_key

    _with_fake_usage(200, body)


def test_a_stated_daily_limit_outlives_a_restart():
    """
    The provider's refusal outlives a restart, so the adopted limit must too.
    Otherwise the UI reports a budget the API has already refused, and the next
    run spends a request rediscovering it.
    """
    from app.index.ratelimit import DailyQuotaReached, RateLimiter

    def body(store):
        refused = RateLimiter()
        refused.acquire(MODEL)
        refused.note_throttled(GEMINI_FLASH_PER_DAY_429, MODEL)   # limit: 20
        assert refused._budget(MODEL).stated_limit == 20

        fresh = RateLimiter()
        try:
            fresh.acquire(MODEL)
        except DailyQuotaReached:
            pass
        else:
            raise AssertionError("forgot a limit the API stated today")

    _with_fake_usage(200, body)


def test_a_stated_limit_never_raises_the_configured_cap():
    """
    Adopted downwards only. A 429 quoting a *higher* number than the user
    configured must not overrule their setting -- and a stale row must not
    silently widen the budget.
    """
    from datetime import date

    from app.index.ratelimit import DailyQuotaReached, RateLimiter

    def body(store):
        limiter = RateLimiter()
        limiter.acquire(MODEL)
        store.conn.execute(
            "INSERT INTO api_limit_by_key VALUES (?, ?, ?, ?) "
            "ON CONFLICT (day, model, credential) DO UPDATE SET "
            "limit_value = excluded.limit_value",
            [date.today(), MODEL, _current_credential(), 5000],
        )

        fresh = RateLimiter()
        fresh.acquire(MODEL)     # 2nd of a configured 3
        fresh.acquire(MODEL)     # 3rd
        try:
            fresh.acquire(MODEL)
        except DailyQuotaReached:
            pass
        else:
            raise AssertionError("a stale stated limit widened the configured cap")

    _with_fake_usage(3, body)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
