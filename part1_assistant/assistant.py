"""
The grounded document assistant.

The flow for every question:

    question
      -> BM25 retrieval over the three documents (retrieval.py)
      -> deterministic conflict check against pricing.json's known_conflicts
      -> LLM, given ONLY the retrieved context, with one tool: calculate_price
      -> if the LLM calls the tool, calculator.py (pure Python) does the arithmetic
      -> LLM writes the answer as JSON {answer, sources, status}
      -> deterministic guardrails fix up anything the model got wrong

Three design choices worth defending:

1. The LLM never does arithmetic. It extracts parameters ("2-bed", "Block B", "floor
   15", "corner", "Margalla-facing") and calls calculate_price; Python computes. Models
   are bad at multi-step percentage maths, and a wrong price costs a sale.

2. The LLM only ever sees the retrieved chunks, never the whole document set and never
   its own background knowledge, and the prompt tells it to refuse rather than fill
   gaps. That is what stops it inventing a rental yield.

3. Guardrails do not trust the model to follow instructions. An answer with no source
   is downgraded, and a known conflict is forced into status="conflict" with both
   values and both sources in code, not by prompting alone.
"""

import json
import re

import calculator
import llm
from retrieval import Retriever

# One retriever, built once (it parses the documents and builds the BM25 index).
_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


# The four statuses the assistant can return. The last three are the cases the brief
# says matter most.
STATUSES = ("answered", "not_in_documents", "conflict", "unconfirmed")

SYSTEM_PROMPT = """You are the internal sales assistant for MGC Developments, used by \
salespeople who are about to quote a customer. A confident wrong answer costs MGC a \
sale; "I don't have that, ask the marketing manager" costs nothing.

RULES, in order of importance:

1. GROUNDING. Answer ONLY from the CONTEXT block in the user message. The context is \
extracts from three internal documents. You have no other knowledge of MGC Aurora \
Heights. If the context does not contain the answer, say so - never guess, never fill \
a gap from general knowledge about Pakistani real estate, never estimate a number.

2. SOURCES. Every answer must cite the document and section it came from, copied \
exactly as they appear in the DOCUMENT: and SECTION: labels of the context. An answer \
without a source is worthless. Cite only sections you actually used.

3. NO ARITHMETIC. You are not allowed to calculate prices, premiums, discounts or \
instalments yourself, not even simple ones. If a question needs any arithmetic, call \
the calculate_price tool and report the numbers it returns, unchanged. If the tool \
returns caveats, pass them on to the salesperson.

4. STATUS. Set exactly one status:
   - "answered": the context answers the question.
   - "not_in_documents": the documents do not contain this information. Say plainly \
that you do not have it and that the salesperson should ask the marketing manager. If \
a document explicitly says MGC does not publish this, cite that section as the source.
   - "conflict": two documents give DIFFERENT values for the same thing. Do NOT pick \
one and do not average them. Give both values, each with its own document and section, \
and tell the salesperson to confirm with the marketing manager before quoting.
   - "unconfirmed": the documents mention the thing but state it is not yet confirmed, \
ongoing, applied for, or subject to approval. Say it is unconfirmed and cite where the \
document says so.

5. OUTPUT. Reply with a single JSON object and nothing else - no prose before or after, \
no markdown code fences:

{"answer": "<plain English for the salesperson, including the numbers the tool \
returned>", "sources": [{"document": "<exact document name>", "section": "<exact \
section name>"}], "status": "<one of answered|not_in_documents|conflict|unconfirmed>"}
"""

# The one tool the model gets. The JSON Schema mirrors calculator.quote()'s signature,
# so the model's job is reduced to filling in a form.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculate_price",
        "description": (
            "Compute the price of an MGC Aurora Heights unit: base price, stacked "
            "location premiums, stacked discounts, the payment schedule and the amount "
            "due at booking. Returns a step-by-step breakdown with the source of every "
            "number. Use this for ANY question involving arithmetic - never calculate "
            "yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "unit_type": {
                    "type": "string",
                    "enum": calculator.list_unit_types(),
                    "description": "Unit type exactly as the price list names it.",
                },
                "block": {
                    "type": "string",
                    "enum": ["A", "B"],
                    "description": (
                        "Which block. Omit only if the customer did not say; it will be "
                        "inferred when the unit type exists in one block only."
                    ),
                },
                "floor": {
                    "type": "integer",
                    "description": (
                        "Floor number 1-22. Omit if the customer did not say a floor - "
                        "do not guess one."
                    ),
                },
                "corner": {"type": "boolean", "description": "True for a corner unit (+3%)."},
                "margalla_facing": {
                    "type": "boolean",
                    "description": "True for a Margalla-facing unit (+6%).",
                },
                "discounts": {
                    "type": "array",
                    "description": "Every discount the buyer qualifies for.",
                    "items": {
                        "type": "string",
                        "enum": ["full_cash", "fifty_percent_upfront",
                                 "overseas_pakistani", "second_unit"],
                    },
                },
                "additional_parking_slots": {
                    "type": "integer",
                    "description": "Extra parking slots beyond the one mandatory slot.",
                },
            },
            "required": ["unit_type"],
        },
    },
}]


def _infer_block(unit_type):
    """If the salesperson didn't say which block, infer it when it is unambiguous.

    "3-Bed Executive" only exists in Block B, so asking back would be pedantic. A
    "2-Bed Standard" exists in both blocks at different prices, so we refuse to guess.
    """
    blocks = [b for b in ("A", "B") if unit_type in calculator.list_unit_types(b)]
    if len(blocks) == 1:
        return blocks[0]
    return None


def _run_calculator(args):
    """Execute the tool call. Returns (payload_for_the_model, quote_dict_or_None).

    Any CalculationError (impossible unit, unknown discount, both payment styles) is
    returned to the model as an error payload rather than raised, so the model can tell
    the salesperson what is wrong instead of the request 500-ing.
    """
    args = dict(args or {})
    unit_type = args.get("unit_type")
    block = (args.get("block") or "").strip().upper() or None

    if not unit_type:
        return {"error": "unit_type is required."}, None

    if block is None:
        block = _infer_block(unit_type)
        if block is None:
            return {
                "error": (
                    f"A {unit_type} is sold in both Block A and Block B at different "
                    "prices. Ask the customer which block before quoting."
                )
            }, None

    try:
        q = calculator.quote(
            unit_type=unit_type,
            block=block,
            floor=args.get("floor"),
            corner=bool(args.get("corner", False)),
            margalla_facing=bool(args.get("margalla_facing", False)),
            discounts=args.get("discounts") or [],
            additional_parking_slots=int(args.get("additional_parking_slots") or 0),
        )
    except calculator.CalculationError as exc:
        return {"error": str(exc)}, None

    # Hand the model a compact summary - the numbers and their sources - not the whole
    # nested dict. The full dict goes back to the caller (and the web page) separately.
    payload = {
        "unit": f"{q['unit_type']}, Block {q['block']}",
        "currency": q["currency"],
        "base_price": q["base_price"],
        "premiums_applied": [f"{p['label']} +{p['pct']}% = {p['amount']:,.0f}" for p in q["premiums"]],
        "premium_total_pct": q["premium_pct_total"],
        "list_price_after_premiums": q["list_price"],
        "discounts_applied": [f"{d['label']} -{d['pct']}% = {d['amount']:,.0f}" for d in q["discounts"]],
        "price_after_discounts": q["net_price"],
        "due_at_booking": q["due_at_booking"],
        "due_at_booking_rule": f"{q['payment_schedule'][0]['pct']}% token under the {q['payment_plan']}",
        "payment_schedule": [
            {"milestone": m["milestone"], "pct": m["pct"], "amount": m["amount"]}
            for m in q["payment_schedule"]
        ],
        "fifty_percent_upfront_amount": q["upfront_50_percent"],
        "other_charges_not_included_in_price": [
            {"label": c["label"], "amount": c["amount"]} for c in q["other_charges"]
        ],
        "caveats_you_must_pass_on": q["caveats"],
        "sources": sorted({
            f"{s['source']['document']} | {s['source']['section']}" for s in q["steps"]
        }),
    }
    return payload, q


def _find_conflict(question, hits):
    """Deterministic conflict detection.

    pricing.json carries a hand-built `known_conflicts` registry: topics where two
    documents state different values (today that is the transfer fee: 2% in the price
    list vs 2.5% in the booking FAQ). We flag a conflict if the topic's keywords appear
    in the question or in any retrieved chunk.

    This is the deterministic safety net. The prompt also asks the model to spot
    conflicts generally, so a disagreement that is not in the registry can still be
    caught - but anything in the registry cannot be silently resolved, whatever the
    model does.
    """
    haystack = (question + " " + " ".join(h["text"] for h in hits)).lower()
    question_lower = question.lower()
    for conflict in calculator.PRICING["known_conflicts"]:
        for kw in conflict["keywords"]:
            # Require the topic in the QUESTION (so an unrelated question that happens
            # to retrieve the Transfers section isn't derailed), or in the question plus
            # the retrieved text if the question used a near-synonym.
            if kw in question_lower or (kw in haystack and kw.split()[0] in question_lower):
                return conflict
    return None


def _extract_json(text):
    """Pull the JSON object out of the model's reply.

    Models sometimes wrap JSON in ```json fences or add a sentence around it, so we
    strip fences and fall back to the outermost {...}. Returns None if nothing parses.
    """
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalise_sources(raw):
    """Coerce whatever the model produced into a clean [{document, section}] list."""
    out = []
    for s in raw or []:
        if isinstance(s, dict):
            doc = str(s.get("document") or s.get("doc") or "").strip()
            sec = str(s.get("section") or "").strip()
        else:
            # Occasionally a model returns "Doc | Section" strings instead of objects.
            parts = str(s).split("|")
            doc = parts[0].strip()
            sec = parts[1].strip() if len(parts) > 1 else ""
        if doc and {"document": doc, "section": sec} not in out:
            out.append({"document": doc, "section": sec})
    return out


def answer(question, k=5):
    """Answer one question. Always returns a dict; never raises for a bad answer.

    Returns:
        {
          question, answer, status, sources: [{document, section}],
          calculation: quote dict or None, calculation_text: str or None,
          retrieved: [{document, section, score}], provider, notes: [str]
        }

    Raises llm.MissingAPIKey / llm.LLMError if the provider cannot be reached - Part 4
    catches those and shows the error while the lead-scoring form keeps working.
    """
    retriever = get_retriever()
    context, hits = retriever.as_context(question, k=k)
    notes = []

    user_parts = [f"CONTEXT (extracts from MGC's internal documents):\n\n{context}"]

    # If this question touches a known cross-document disagreement, say so loudly in the
    # prompt, with both values, before the model answers.
    conflict = _find_conflict(question, hits)
    if conflict:
        both = "\n".join(
            f'  - "{v["value"]}" from {v["document"]} -> {v["section"]} ("{v["quote"]}")'
            for v in conflict["values"]
        )
        user_parts.append(
            f"CONFLICT ALERT: the documents disagree about the {conflict['topic']}:\n{both}\n"
            f"You MUST use status \"conflict\", report BOTH values with BOTH sources, and "
            f"tell the salesperson: {conflict['action']}"
        )

    user_parts.append(f"SALESPERSON'S QUESTION: {question}")
    messages = [{"role": "user", "content": "\n\n".join(user_parts)}]

    quote = None
    # Up to 3 round trips: enough for retrieve -> tool call -> final answer, with one
    # spare if the model needs a second calculation. A hard cap stops any runaway loop.
    for _ in range(3):
        reply = llm.chat(SYSTEM_PROMPT, messages, tools=TOOLS)
        messages.append(reply["assistant_message"])

        if not reply["tool_calls"]:
            break

        for tc in reply["tool_calls"]:
            if tc["name"] != "calculate_price":
                payload = {"error": f"Unknown tool {tc['name']!r}."}
            else:
                payload, q = _run_calculator(tc["arguments"])
                if q is not None:
                    quote = q
                if "error" in payload:
                    notes.append(f"Calculator rejected the request: {payload['error']}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(payload, default=str),
            })

    parsed = _extract_json(reply["text"])
    if parsed is None:
        # The model didn't return usable JSON. Don't pretend it did: hand back whatever
        # it said, flagged as unconfirmed so nobody quotes it to a customer.
        notes.append("The model did not return valid JSON; returning its raw text, unverified.")
        result = {
            "answer": (reply["text"] or "").strip() or "No answer was produced.",
            "status": "unconfirmed",
            "sources": [],
        }
    else:
        result = {
            "answer": str(parsed.get("answer", "")).strip(),
            "status": str(parsed.get("status", "")).strip(),
            "sources": _normalise_sources(parsed.get("sources")),
        }

    # ---- deterministic guardrails: do not trust the model to have obeyed ----------

    if result["status"] not in STATUSES:
        notes.append(f"Model returned an unrecognised status {result['status']!r}; treating as unconfirmed.")
        result["status"] = "unconfirmed"

    if conflict:
        # Force the conflict, whatever the model decided.
        if result["status"] != "conflict":
            # The model tried to resolve the disagreement (e.g. answered "2%"). Its
            # answer is not safe to show, so replace it with a deterministic statement
            # of both values built from pricing.json.
            notes.append(
                "Forced status=conflict and rewrote the answer: this topic is a known "
                "cross-document disagreement and the model gave a single value."
            )
            result["status"] = "conflict"
            both = "; ".join(
                f'{v["value"]} according to {v["document"]} ({v["section"]})'
                for v in conflict["values"]
            )
            result["answer"] = (
                f"The documents disagree about the {conflict['topic']}: {both}. "
                f"{conflict['action']}"
            )
        # Make sure both sources are cited even if the model only cited one.
        for v in conflict["values"]:
            src = {"document": v["document"], "section": v["section"]}
            if src not in result["sources"]:
                result["sources"].append(src)
        if "marketing manager" not in result["answer"].lower():
            result["answer"] = result["answer"].rstrip() + " Confirm with the marketing manager before quoting a customer."

    # "An answer without a source is wrong" - the brief. So an answered/conflict reply
    # with no citation is downgraded rather than shown as fact.
    if result["status"] in ("answered", "conflict") and not result["sources"]:
        notes.append("Downgraded to unconfirmed: the model gave an answer with no source.")
        result["status"] = "unconfirmed"

    return {
        "question": question,
        "answer": result["answer"],
        "status": result["status"],
        "sources": result["sources"],
        "calculation": quote,
        "calculation_text": calculator.format_breakdown(quote) if quote else None,
        "retrieved": [
            {"document": h["document"], "section": h["section"], "score": h["score"]} for h in hits
        ],
        "provider": llm.describe(),
        "notes": notes,
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "What's the transfer fee?"
    try:
        r = answer(question)
    except llm.MissingAPIKey as exc:
        print("ERROR:", exc)
        sys.exit(1)
    print(f"Q: {r['question']}")
    print(f"STATUS: {r['status']}")
    print(f"A: {r['answer']}")
    for s in r["sources"]:
        print(f"   source: {s['document']} -> {s['section']}")
    if r["calculation_text"]:
        print(r["calculation_text"])
    for n in r["notes"]:
        print(f"   note: {n}")
