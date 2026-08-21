"""
The agent loop: the model decides which tool to call, we run it, repeat.

Provider-neutral. The conversation is held in OpenAI's message shape and handed
to whichever provider is configured -- Gemini's SDK, or any endpoint speaking
chat-completions (Groq, xAI, OpenRouter, a local Ollama). See agent/llm.py for
the translation.

Every tool call is recorded and returned alongside the answer. That trace is
not debug output -- it is the point. When the bot says "Rohit sent 4,182
messages" you can read the exact SQL it ran and check it yourself, which is the
difference between a statistic and a guess.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..config import settings
from ..db import get_cursor, get_meta
from .llm import (LLMError, build_provider, is_too_large,
                  openai_tool_schema)
from .prompts import build_system_prompt
from .tools import TOOL_DECLARATIONS, build_tools


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: Any = None
    error: str | None = None
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": _truncate_result(self.result),
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentAnswer:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    error: str | None = None
    provider: str = ""

    def as_dict(self) -> dict:
        return {
            "answer": self.text,
            "tool_calls": [tc.as_dict() for tc in self.tool_calls],
            "steps": self.steps,
            "error": self.error,
            "provider": self.provider,
        }


def _truncate_result(result: Any, max_rows: int = 30) -> Any:
    """Keep traces readable in the UI without hiding that data was cut."""
    if not isinstance(result, dict):
        return result
    trimmed = dict(result)
    if isinstance(trimmed.get("rows"), list) and len(trimmed["rows"]) > max_rows:
        trimmed["rows"] = trimmed["rows"][:max_rows]
        trimmed["_display_note"] = (
            f"showing {max_rows} of {result['row_count']} rows")
    if isinstance(trimmed.get("results"), list) and len(trimmed["results"]) > max_rows:
        trimmed["results"] = trimmed["results"][:max_rows]
    return trimmed


# Questions whose answer is a judgement rather than a lookup. These have to
# search, often read context around a hit, and weigh what came back -- so they
# get the widest budget. Cutting one of these off does not produce a shorter
# answer, it produces no answer, which is the worst outcome available.
_REASONING_MARKERS = (
    "decide", "decided", "decision", "agree", "agreed", "conclude",
    "conclusion", "outcome", "resolve", "settle", "why", "what happened",
    "argument", "argue", "argued", "fight", "fought", "discuss", "discussion",
    "summar", "explain", "opinion", "plan for", "planning", "context",
    "plan about", "think about", "feel about", "plan on",
)

# Questions with a single computable answer. These are one run_sql call in
# practice; extra room just invites the model to keep querying after it already
# has the rows, which is exactly how a 2-second answer became 77 seconds.
_DIRECT_MARKERS = (
    "how many", "how much", "how often", "count", "average", "total",
    "most", "least", "top ", "rank", "busiest", "longest", "shortest",
    "fastest", "slowest", "percentage", "percent", "who sent", "what time",
    "when did", "list ", "highest", "lowest", "number of",
)


# Sent when the model tries to answer without having called a single tool. It
# does not have the chat in context, so such an answer can only come from its
# own world knowledge -- and it reads as authoritative. Asked to explain a term
# that appears only inside the chat, it produced a confident essay from general
# knowledge, complete with a formatted table, and never touched the archive.
# That is the one failure this whole app exists to prevent.
_NO_TOOL_NUDGE = (
    "You have not looked at the archive. You do not have this chat in your "
    "context, so anything you write now would be from general knowledge, not "
    "from these messages -- which is exactly wrong for this question. Call a "
    "tool first: search_chat for what was said, run_sql for anything "
    "countable. If the archive turns out not to mention it, say that plainly "
    "and do not explain the term from general knowledge."
)


def step_budget(question: str) -> int:
    """
    How many tool steps this question is allowed.

    Every step re-sends the whole conversation, so against a per-minute token
    allowance the budget is the single biggest lever on how long an answer
    takes. A fixed budget has to be wrong in one direction or the other: at 4,
    a "what did we decide about X" question ran out of steps and answered
    nothing at all; at 8, "who starts conversations most often" -- one SQL
    query -- was given room to wander through three more.

    Reasoning is checked first on purpose. When a question looks like both, the
    generous reading is the safer error: too many steps is slow, too few is
    empty.
    """
    text = (question or "").lower()
    if any(marker in text for marker in _REASONING_MARKERS):
        return max(1, settings.agent_max_steps_reasoning)
    if any(marker in text for marker in _DIRECT_MARKERS):
        return max(1, settings.agent_max_steps_direct)
    return max(1, settings.agent_max_steps)


def _call_key(name: str, args: dict) -> str:
    try:
        return name + "|" + json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + "|" + str(args)


def _execute(name: str, args: dict, tools: dict,
             seen: dict | None = None) -> ToolCall:
    call = ToolCall(name=name, arguments=args)

    # A model that gets an unhelpful result often reruns the identical call,
    # burning the step budget three or four times over on a query that cannot
    # start returning rows. Hand back the first result with a nudge instead.
    if seen is not None:
        key = _call_key(name, args)
        if key in seen:
            previous = seen[key]
            call.result = {
                **(previous if isinstance(previous, dict) else
                   {"result": previous}),
                "_repeat": "You already ran this exact call and got this same "
                           "result. Repeating it will not change anything -- "
                           "try a different approach, or answer with what you "
                           "have and say what is missing.",
            }
            call.duration_ms = 0
            return call

    fn = tools.get(name)
    if fn is None:
        call.error = f"Unknown tool: {name}"
        call.result = {"error": call.error}
        return call

    started = time.time()
    try:
        call.result = fn(**args)
    except TypeError as exc:
        call.error = f"Bad arguments for {name}: {exc}"
        call.result = {"error": call.error}
    except Exception as exc:  # noqa: BLE001 - reported back so the model retries
        call.error = str(exc)
        call.result = {"error": call.error}
    call.duration_ms = int((time.time() - started) * 1000)
    if seen is not None:
        seen[_call_key(name, args)] = call.result
    return call


def _tool_result_payload(result: Any) -> str:
    """Serialise a tool result for the model, capped so one call cannot fill
    the whole budget."""
    try:
        text = json.dumps(_truncate_result(result), default=str)
    except (TypeError, ValueError):
        text = str(result)
    limit = settings.tool_result_max_chars
    if len(text) > limit:
        return text[:limit] + '..."[truncated]"'
    return text


def _summarise_tool_result(content: str) -> str:
    """
    Replace a bulky old tool result with just its shape.

    Once the model has moved on, the rows themselves are dead weight; what it
    still needs is the knowledge that the call happened and roughly what came
    back.
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content[:200] + " [earlier result trimmed]"

    if not isinstance(payload, dict):
        return "[earlier result trimmed]"

    keep = {k: payload[k] for k in
            ("row_count", "result_count", "total_in_window", "error", "sql")
            if k in payload}
    keep["_trimmed"] = "full result omitted to stay within the token budget"
    return json.dumps(keep, default=str)[:600]


# Results at or under this size are never summarised: keeping them is cheap
# and they are frequently the finding itself rather than supporting detail.
_KEEP_WHOLE_BELOW = 1_200


def _message_size(message: dict) -> int:
    """
    Bytes a message contributes to the request.

    Counts the serialised tool_calls too: an assistant turn requesting three
    queries carries real weight, and ignoring it made the budget optimistic
    exactly when the conversation was largest.
    """
    size = len(message.get("content") or "")
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        size += len(str(function.get("arguments") or ""))
        size += len(str(function.get("name") or ""))
    return size


def _trim_conversation(messages: list[dict],
                       budget: int | None = None) -> list[dict]:
    """
    Keep the request inside the provider's tokens-per-minute allowance.

    Free tiers limit tokens per minute, not just request count, and an agent
    conversation grows with every tool result -- so it is the third or fourth
    call in a run that trips the limit, right when the model is close to an
    answer. Recent results stay whole because they are what the model is
    reasoning about; older ones shrink to their shape.
    """
    budget = budget or settings.context_budget_chars

    # The budget covers the conversation only. The system prompt is fixed and
    # untrimmable -- it carries the schema needed to write correct SQL -- so
    # counting it here would make the setting mean something different at every
    # archive size, and leave almost nothing for actual tool results.
    total = sum(_message_size(m) for m in messages
                if m.get("role") != "system")
    if total <= budget:
        return messages

    trimmed = list(messages)
    running = 0
    newest_tool_seen = False

    for i in range(len(trimmed) - 1, -1, -1):
        message = trimmed[i]
        if message.get("role") == "system":
            continue

        size = _message_size(message)
        running += size

        if message.get("role") != "tool":
            continue

        if not newest_tool_seen:
            # The most recent result is what the model is actually reasoning
            # about. If it alone overruns the budget, cut it down rather than
            # replacing it with a stub -- discarding it entirely would leave
            # nothing to answer from.
            newest_tool_seen = True
            if size > budget:
                content = message.get("content") or ""
                trimmed[i] = {
                    **message,
                    "content": content[:budget] + '..."[truncated to fit]"',
                }
            continue

        # Only bulky results are worth trimming. A small one costs almost
        # nothing to keep and is often the whole answer: "which day had the
        # most messages" comes back as a single two-value row, and summarising
        # that away left the model reporting that its own finding had been
        # trimmed out.
        if running > budget and size > _KEEP_WHOLE_BELOW:
            trimmed[i] = {**message,
                          "content": _summarise_tool_result(
                              message.get("content") or "")}
    return trimmed


def _complete(provider, messages: list[dict], schema) -> Any:
    """
    Ask the model, shrinking the conversation if the request is refused as
    too large.

    Providers cap a single request as well as the per-minute rate, and the cap
    varies by model and account. Rather than guessing a budget that fits
    everywhere, start at the configured one and halve it on rejection -- which
    self-corrects regardless of whether the estimate was right.
    """
    budget = settings.context_budget_chars
    last: Exception | None = None

    for attempt in range(3):
        try:
            return provider.complete(_trim_conversation(messages, budget), schema)
        except LLMError as exc:
            last = exc
            if not is_too_large(str(exc)) or attempt == 2:
                raise
            budget = max(1_500, budget // 2)
            print(f"[agent] request refused as too large; retrying with a "
                  f"{budget}-character context budget")
    raise last  # pragma: no cover


def _build_messages(system_prompt: str, question: str,
                    history: list[dict] | None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-10:]:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": question})
    return messages


def _prepare(question: str, history: list[dict] | None, archive):
    """Shared setup for both the blocking and streaming entry points."""
    conn = get_cursor(archive)
    tools = build_tools(conn)

    overview_raw = get_meta("overview", conn)
    overview = json.loads(overview_raw) if overview_raw else {}
    system_prompt = build_system_prompt(overview, settings.session_gap_hours)

    provider = build_provider()
    schema = openai_tool_schema(TOOL_DECLARATIONS)
    messages = _build_messages(system_prompt, question, history)
    return provider, tools, schema, messages


def _record_calls(response, tools, messages, answer, seen) -> None:
    """Run the requested tools and append the exchange to the conversation."""
    messages.append({
        "role": "assistant",
        "content": response.text or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in response.tool_calls
        ],
    })

    for requested in response.tool_calls:
        call = _execute(requested.name, requested.arguments, tools, seen)
        answer.tool_calls.append(call)
        messages.append({
            "role": "tool",
            "tool_call_id": requested.id,
            "name": requested.name,
            "content": _tool_result_payload(call.result),
        })


def ask(question: str, history: list[dict] | None = None,
        archive=None) -> AgentAnswer:
    """Answer one question, running tools until the model is done."""
    answer = AgentAnswer()

    try:
        provider, tools, schema, messages = _prepare(question, history, archive)
    except LLMError as exc:
        answer.error = str(exc)
        return answer

    answer.provider = provider.name
    seen: dict = {}
    budget = step_budget(question)
    nudged = False

    for step in range(budget):
        answer.steps = step + 1
        try:
            response = _complete(provider, messages, schema)
        except Exception as exc:  # noqa: BLE001
            answer.error = f"Model call failed: {exc}"
            return answer

        if not response.wants_tools:
            # An answer with no tool call behind it is a guess dressed as a
            # fact. Push back once; the model almost always searches when told.
            if not answer.tool_calls and not nudged:
                nudged = True
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue

            answer.text = response.text
            if not answer.text:
                answer.error = "Model returned an empty answer."
            return answer

        _record_calls(response, tools, messages, answer, seen)

    # Out of tool steps. Do not throw away what was gathered: the model has
    # usually found most of the answer by now and only failed to stop calling
    # tools. Ask once more with the tools removed so it must reply in words.
    forced = _force_answer(provider, messages)
    if forced:
        answer.text = forced
    else:
        answer.error = (
            f"Stopped after {budget} tool steps without reaching an answer. "
            "Try asking something more specific."
        )
    return answer


def _force_answer(provider, messages: list[dict]) -> str:
    """
    Final pass with no tools offered, so the model must answer or say it cannot.

    The conversation is *flattened* first: system prompt, then one user turn
    holding the question and the tool results as plain text. Sending the real
    history here fails in a way that costs the user their whole answer -- with
    assistant tool-call turns in the transcript the model emits another tool
    call, and Groq rejects the entire request with
    `400 tool_use_failed: "Tool choice is none, but model called a tool"`.
    That happened on a question whose *first* SQL call had already returned the
    answer: four steps of work, then an empty reply.

    Flattening also drops the tool schema from the request, so this last call is
    the cheapest one of the turn rather than the largest.
    """
    system = next((m for m in messages if m.get("role") == "system"), None)

    question = ""
    results: list[str] = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "user" and not question:
            question = content
        elif role == "tool" and content:
            name = message.get("name") or "tool"
            results.append(f"Result of {name}:\n{content}")

    # Budget the flattened text. Concatenating every result unconditionally is
    # how this call reached 14,629 tokens against an 8,000/minute ceiling -- and
    # because it is a single message, the shrink-and-retry in _complete could not
    # cut it down either, so the whole answer was lost at the last step. Newest
    # results are kept: they are what the model was working on when it ran out.
    budget = max(1_000, settings.context_budget_chars)
    kept: list[str] = []
    used = len(question)
    for block in reversed(results):
        if used + len(block) > budget:
            room = budget - used
            if room > 300:
                kept.append(block[:room] + '... [truncated to fit]')
            break
        kept.append(block)
        used += len(block)

    transcript = [f"Question: {question}"] + list(reversed(kept))

    closing = ([system] if system else []) + [{
        "role": "user",
        "content": "\n\n".join(transcript) + "\n\n" + (
            "Answer now, using only what the tool results above already "
            "contain. Give as much of the actual answer as you can -- figures, "
            "quotes, specifics. If part of it is genuinely unavailable, say so "
            "in one short sentence rather than a section. Do not invent "
            "anything and do not mention running out of tool calls.\n"
            # No tools are offered on this call, and Groq rejects the whole
            # request with a 400 tool_use_failed if the model emits a tool call
            # anyway -- which gpt-oss does. Saying so plainly is cheaper than
            # discovering it: the failure costs another request and its rate
            # limit wait, at the very end of an already slow answer.
            "Reply with prose only. Do not call any tool."
        ),
    }]
    try:
        response = _complete(provider, closing, None)
        # Only here: no tools were offered and the model was told to write the
        # final answer, so a reply that landed in `reasoning` is the answer
        # rather than private monologue. On a normal turn this would leak the
        # model's thinking to the user.
        return response.text or response.reasoning
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] forced answer failed: {exc}")
        return ""


def ask_stream(question: str, history: list[dict] | None = None,
               archive=None) -> Iterator[dict]:
    """
    Same as ask(), but yields events so the UI can show work as it happens.

    Events: {"type": "tool_call"|"tool_result"|"answer"|"error", ...}
    """
    answer = AgentAnswer()

    try:
        provider, tools, schema, messages = _prepare(question, history, archive)
    except LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "provider", "name": provider.name}
    seen: dict = {}
    budget = step_budget(question)
    nudged = False

    for step in range(budget):
        try:
            response = _complete(provider, messages, schema)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"Model call failed: {exc}"}
            return

        if not response.wants_tools:
            if not answer.tool_calls and not nudged:
                nudged = True
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue

            yield {"type": "answer", "text": response.text,
                   "tool_calls": [tc.as_dict() for tc in answer.tool_calls],
                   "steps": step + 1}
            return

        messages.append({
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name,
                              "arguments": json.dumps(c.arguments)}}
                for c in response.tool_calls
            ],
        })

        for requested in response.tool_calls:
            yield {"type": "tool_call", "name": requested.name,
                   "arguments": requested.arguments}
            call = _execute(requested.name, requested.arguments, tools, seen)
            answer.tool_calls.append(call)
            yield {"type": "tool_result", **call.as_dict()}
            messages.append({
                "role": "tool",
                "tool_call_id": requested.id,
                "name": requested.name,
                "content": _tool_result_payload(call.result),
            })

    forced = _force_answer(provider, messages)
    if forced:
        yield {"type": "answer", "text": forced,
               "tool_calls": [tc.as_dict() for tc in answer.tool_calls],
               "steps": budget,
               "note": "Ran out of tool calls; answered from what was gathered."}
    else:
        yield {"type": "error",
               "message": f"Stopped after {budget} tool steps without "
                          "reaching an answer."}
