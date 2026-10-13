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
        "default_model": "gemini-2.5-flash",
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


def model_name():
    provider = provider_name()
    if provider not in PROVIDERS:
        raise LLMError(
            f"LLM_PROVIDER={provider!r} is not supported. Use one of: {', '.join(PROVIDERS)}."
        )
    return (os.getenv("MODEL") or "").strip() or PROVIDERS[provider]["default_model"]


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
    return f"{provider_name()} / {model_name()}"


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

    Differences we have to bridge:
      - a tool call is an assistant "tool_use" content block, not a `tool_calls` field;
      - a tool result is a USER message with a "tool_result" block, not role="tool";
      - Anthropic takes `system` as its own argument (handled in chat()).
    """
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }],
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"] or "{}"),
                })
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def chat(system, messages, tools=None, max_tokens=2000):
    """Send one request and return a normalised reply.

    Args:
        system: the system prompt (a string).
        messages: history in OpenAI shape (see module docstring).
        tools: tool specs in OpenAI shape, or None.

    Returns:
        {
          "text": str,                  # the assistant's text, "" if it only called tools
          "tool_calls": [ {"id": str, "name": str, "arguments": dict} ],
          "assistant_message": dict,    # OpenAI-shaped, append it straight to `messages`
        }
    """
    provider = provider_name()
    key = api_key()
    model = model_name()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=_to_anthropic_messages(messages),
                **({"tools": _to_anthropic_tools(tools)} if tools else {}),
            )
        except Exception as exc:  # network, bad model name, rate limit, ...
            raise LLMError(f"{provider} call failed (MODEL={model}): {exc}") from exc

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            {"id": b.id, "name": b.name, "arguments": b.input}
            for b in resp.content if b.type == "tool_use"
        ]
        # Rebuild the reply in OpenAI shape so the caller stays provider-agnostic.
        assistant_message = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_calls
            ]
        return {"text": text, "tool_calls": tool_calls, "assistant_message": assistant_message}

    # ---- gemini and openai: identical code, different base_url + key ----------
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    client = OpenAI(api_key=key, base_url=cfg["base_url"]) if cfg["base_url"] else OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            # The system prompt is just the first message in this API.
            messages=[{"role": "system", "content": system}] + messages,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        )
    except Exception as exc:
        raise LLMError(f"{provider} call failed (MODEL={model}): {exc}") from exc

    choice = resp.choices[0].message
    tool_calls = []
    for tc in (choice.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            # A model can emit malformed JSON arguments. Surface it as a tool error
            # rather than crashing; assistant.py feeds the error back to the model.
            args = {"_malformed_arguments": tc.function.arguments}
        tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

    assistant_message = {"role": "assistant", "content": choice.content or ""}
    if choice.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in choice.tool_calls
        ]
    return {"text": choice.content or "", "tool_calls": tool_calls,
            "assistant_message": assistant_message}


if __name__ == "__main__":
    print("provider/model:", describe())
    try:
        api_key()
        print("API key: found")
    except MissingAPIKey as exc:
        print("API key: MISSING ->", exc)
