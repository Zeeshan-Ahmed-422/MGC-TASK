"""
One thin wrapper around three LLM providers, so assistant.py contains no provider
branching at all.

Switching provider is one line in .env:

    LLM_PROVIDER=gemini     MODEL=gemini-2.5-flash    GOOGLE_API_KEY=...   (default)
    LLM_PROVIDER=openai     MODEL=gpt-4o-mini         OPENAI_API_KEY=...
    LLM_PROVIDER=anthropic  MODEL=claude-sonnet-5-5   ANTHROPIC_API_KEY=...

Why only ~130 lines for three providers: Google exposes an OpenAI-compatible endpoint,
so `gemini` and `openai` are literally the same code with a different base_url and key.
Only Anthropic needs its own branch, because its tool-calling message shape differs.

The canonical message/tool format used everywhere in this project is the OpenAI one
(two of the three providers speak it natively). The Anthropic branch translates to and
from it, so the rest of the codebase never sees the difference.
"""

import json
import os
import re
import time

from dotenv import load_dotenv

# Load .env from the repo root (one level up from this file) regardless of the current
# working directory, so `python part1_assistant/run_test_questions.py` and
# `python app.py` both pick up the same key.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))

# Per-provider: which env var holds the key, which model to use if MODEL is unset, and
# (for the OpenAI-compatible ones) which base URL to talk to.
PROVIDERS = {
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-3.5-flash,gemini-3.1-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "signup": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "base_url": None,  # the SDK's own default
        "signup": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-5-5",
        "base_url": None,
        "signup": "https://console.anthropic.com/settings/keys",
    },
}


class MissingAPIKey(RuntimeError):
    """No API key configured. Part 4 catches this so the page still loads and the lead
    scoring form keeps working while the document assistant shows a clear error."""


class LLMError(RuntimeError):
    """The provider rejected the call (bad model name, rate limit, network, ...)."""


def provider_name():
    return (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()


def model_names():
    """The models to try, in order. MODEL may be a comma-separated fallback chain.

    WHY A CHAIN. Google's free tier caps each model separately - 20 requests a day on
    gemini-3.5-flash when I was building - and the models differ in speed by a lot: when
    3.5-flash has quota it answers in under 2 seconds, and the lite model takes about 10.
    So the default is "fast one first, dependable one second": questions are quick while
    the daily allowance lasts and keep working after it runs out, instead of the whole
    thing dying with a quota error. One model name here is perfectly valid too.
    """
    provider = provider_name()
    if provider not in PROVIDERS:
        raise LLMError(
            f"LLM_PROVIDER={provider!r} is not supported. Use one of: {', '.join(PROVIDERS)}."
        )
    configured = (os.getenv("MODEL") or "").strip() or PROVIDERS[provider]["default_model"]
    return [m.strip() for m in configured.split(",") if m.strip()]


def model_name():
    """The first model in the chain - what we try before any fallback."""
    return model_names()[0]


def api_key():
    """Return the key for the configured provider, or raise MissingAPIKey with a message
    that tells the user exactly which variable to set and where to get a key."""
    provider = provider_name()
    if provider not in PROVIDERS:
        raise LLMError(
            f"LLM_PROVIDER={provider!r} is not supported. Use one of: {', '.join(PROVIDERS)}."
        )
    cfg = PROVIDERS[provider]
    key = (os.getenv(cfg["key_env"]) or "").strip()
    if not key:
        raise MissingAPIKey(
            f"No API key found. Copy .env.example to .env and set {cfg['key_env']}= "
            f"(LLM_PROVIDER is currently {provider!r}). Get a key at {cfg['signup']}."
        )
    return key


def describe():
    """Short string for logs and the web page footer."""
    chain = model_names()
    shown = chain[0] + (f" (falling back to {', '.join(chain[1:])})" if len(chain) > 1 else "")
    return f"{provider_name()} / {shown}"


def _to_anthropic_tools(tools):
    """OpenAI tool spec -> Anthropic tool spec (same JSON Schema, different wrapper)."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"]["parameters"],
        }
        for t in tools or []
    ]


def _to_anthropic_messages(messages):
    """OpenAI-shaped history -> Anthropic-shaped history.

    Only plain user/assistant text messages are ever sent (see the note on tool calls
    in chat()), so this is a simple rename of the content field. Anthropic takes
    `system` as its own argument, which chat() handles.
    """
    return [{"role": m["role"], "content": m.get("content") or ""} for m in messages]


# Failures worth retrying: rate limits, timeouts, and the provider being briefly
# overloaded. Matched on the error text rather than per-SDK exception classes, so one
# list covers all three providers.
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "overloaded",
                      "high demand", "unavailable", "timeout", "timed out")


# Seconds to wait for one provider call before giving up on it and retrying. Short on
# purpose - see _with_retries(). Override with MGC_LLM_TIMEOUT if your key is slower.
REQUEST_TIMEOUT = float(os.getenv("MGC_LLM_TIMEOUT", "25"))


class QuotaExhausted(LLMError):
    """The key's quota for this model is gone for the day - retrying will not help."""


class ModelUnavailable(LLMError):
    """This key cannot use this model at all (retired, or not granted) - try the next."""


# Models found to be out of quota (or unusable) during this process's lifetime. A plain
# set is enough: adding and checking are atomic under the GIL, and the worst case if two
# threads race is one wasted retry.
_EXHAUSTED = set()


def _retry_after_seconds(text):
    """Pull Google's suggested retry delay out of an error body, if there is one.

    Google returns e.g. "'retryDelay': '25s'" for a per-minute limit and something like
    '37145s' (ten hours) when the per-DAY free-tier quota is gone. That number is how we
    tell a short wait from a dead end.
    """
    match = re.search(r"retryDelay'?:?\s*'?(\d+)s", text)
    return int(match.group(1)) if match else None


def _classify(exc):
    """Return 'quota', 'unavailable', 'transient' (worth a retry) or 'permanent'."""
    text = str(exc).lower()
    # A model this key cannot use: retired for new keys, or never granted. Google
    # retired gemini-2.5-flash for new keys while I was building this, which is exactly
    # the case the fallback chain exists for.
    if "404" in text or "not found" in text or "no longer available" in text:
        return "unavailable"
    if "429" in text or "quota" in text:
        delay = _retry_after_seconds(str(exc))
        # A per-day quota, or any suggested wait longer than five minutes, is not
        # something a retry loop can fix - fail fast with a message that says so.
        if "perday" in text.replace("_", "").replace("-", "") or (delay and delay > 300):
            return "quota"
        return "transient"
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


def _with_retries(call, attempts=4, first_delay=1.5):
    """Run `call`, retrying transient provider failures with exponential backoff.

    Google's free tier returns 503 "experiencing high demand" and short 429s often
    enough that one unlucky question would otherwise fail a whole six-question run, so
    this retries with a growing delay (1.5s, 3s, 6s). A bad model name or a bad key is
    not retried, and neither is an exhausted daily quota - both would just fail four
    times and waste the reviewer's time.

    The delays are deliberately short, and REQUEST_TIMEOUT below is deliberately well
    under a minute: the free tier occasionally takes 80 seconds to answer a call that
    normally takes 5, and abandoning that call and asking again is far quicker than
    waiting for it. Fail fast, retry fast.
    """
    delay = first_delay
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            kind = _classify(exc)
            if kind == "quota":
                hours = (_retry_after_seconds(str(exc)) or 0) / 3600
                raise QuotaExhausted(
                    "the free-tier daily quota is used up"
                    + (f" (resets in about {hours:.0f}h)" if hours else "")
                    + "; add another model to MODEL in .env, or switch LLM_PROVIDER"
                ) from exc
            if kind == "unavailable":
                raise ModelUnavailable(
                    "this key cannot use that model (retired or not granted); put a model "
                    "it can use in MODEL in .env"
                ) from exc
            if attempt == attempts or kind == "permanent":
                raise
            print(f"  (provider error, retry {attempt}/{attempts - 1} in {delay:.0f}s: "
                  f"{str(exc)[:110]})")
            time.sleep(delay)
            delay *= 2


def chat(system, messages, tools=None, max_tokens=2000, force_tool=None):
    """Send one request, trying each model in the MODEL chain until one answers.

    A model is skipped only when it cannot serve this request at all - its daily free
    quota is gone, or the name is not one this key can use. Any other failure is already
    retried inside _chat_one_model, and is raised rather than silently falling through,
    because a working model returning an error is not something a different model fixes.
    """
    chain = model_names()
    # Skip models already known to be out of quota, but never skip the last one - if
    # every model is exhausted we still want the real error, not an empty chain.
    usable = [m for m in chain if m not in _EXHAUSTED] or chain[-1:]

    last_error = None
    for position, model in enumerate(usable):
        try:
            reply = _chat_one_model(model, system, messages, tools, max_tokens, force_tool)
            reply["model"] = model
            return reply
        except (QuotaExhausted, ModelUnavailable) as exc:
            last_error = exc
            # Remember it, so the next question does not spend a round trip rediscovering
            # that this model is finished for the day. Forgotten when the process restarts,
            # which is the right granularity for a quota that resets on a timer.
            _EXHAUSTED.add(model)
            if position + 1 < len(usable):
                print(f"  ({model} unavailable, falling back to {usable[position + 1]}: "
                      f"{str(exc)[:90]})")
    raise last_error


def _chat_one_model(model, system, messages, tools, max_tokens, force_tool):
    """Send one request and return a normalised reply.

    Args:
        system: the system prompt (a string).
        messages: plain user/assistant text messages (see the note below).
        tools: tool specs in OpenAI shape, or None.
        force_tool: name of a tool the model MUST call, instead of choosing. Used when a
            question plainly needs the price calculator and the model answered without
            it - the prompt asked nicely, this does not.

    Returns:
        {
          "text": str,                  # the assistant's text, "" if it only called tools
          "tool_calls": [ {"id": str, "name": str, "arguments": dict} ],
        }

    NOTE ON TOOL CALLS - why there is no tool-result message type here.
    The obvious design is the standard agent loop: send the tool's result back as a
    `role="tool"` message and let the model continue the same conversation. I started
    there and Gemini rejected it:

        400 Function call is missing a thought_signature in functionCall parts.

    Gemini attaches an opaque signature to each function call and requires it back when
    you replay that call in the history; the OpenAI-compatible endpoint does not expose
    it, so the replay can never be valid. Rather than special-case one provider, the
    assistant does this instead: first call extracts the parameters, Python computes the
    answer, and then a SECOND, fresh call is made with the computed numbers written into
    the prompt as context. No tool message is ever replayed, so nothing provider-specific
    leaks in here, and the model still cannot do the arithmetic itself - it only ever
    sees numbers Python produced. See assistant.py.
    """
    provider = provider_name()
    key = api_key()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT, max_retries=0)
        try:
            tool_args = {}
            if tools:
                tool_args["tools"] = _to_anthropic_tools(tools)
                if force_tool:
                    tool_args["tool_choice"] = {"type": "tool", "name": force_tool}
            resp = _with_retries(lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=_to_anthropic_messages(messages),
                **tool_args,
            ))
        except (QuotaExhausted, ModelUnavailable):
            raise   # the chain in chat() handles these by trying the next model
        except Exception as exc:  # network, rate limit, ...
            raise LLMError(f"{provider} call failed (MODEL={model}): {exc}") from exc

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            {"id": b.id, "name": b.name, "arguments": b.input}
            for b in resp.content if b.type == "tool_use"
        ]
        return {"text": text, "tool_calls": tool_calls}

    # ---- gemini and openai: identical code, different base_url + key ----------
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    # timeout: a hung request must not block the whole run. max_retries=0: the SDK's own
    # silent retries fought with _with_retries() above, which at least logs what it is
    # doing - one retry policy is easier to reason about than two.
    client_args = {"api_key": key, "timeout": REQUEST_TIMEOUT, "max_retries": 0}
    if cfg["base_url"]:
        client_args["base_url"] = cfg["base_url"]
    client = OpenAI(**client_args)
    try:
        tool_args = {}
        if tools:
            tool_args["tools"] = tools
            tool_args["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
            )
        resp = _with_retries(lambda: client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            # The system prompt is just the first message in this API.
            messages=[{"role": "system", "content": system}] + messages,
            **tool_args,
        ))
    except (QuotaExhausted, ModelUnavailable):
        raise   # the chain in chat() handles these by trying the next model
    except Exception as exc:
        raise LLMError(f"{provider} call failed (MODEL={model}): {exc}") from exc

    choice = resp.choices[0].message
    tool_calls = []
    for tc in (choice.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            # A model can emit malformed JSON arguments. Surface it as a tool error
            # rather than crashing; assistant.py turns it into an error for the user.
            args = {"_malformed_arguments": tc.function.arguments}
        tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

    return {"text": choice.content or "", "tool_calls": tool_calls}


if __name__ == "__main__":
    print("provider/model:", describe())
    try:
        api_key()
        print("API key: found")
    except MissingAPIKey as exc:
        print("API key: MISSING ->", exc)
