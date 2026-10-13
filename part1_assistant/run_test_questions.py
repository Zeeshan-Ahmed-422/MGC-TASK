"""
Runs the six questions from the brief through the assistant and writes test_output.md.

Usage (from the repo root):
    python part1_assistant/run_test_questions.py

The last three questions are the ones that matter most: a cross-document conflict, a
question the documents don't answer, and something the documents say is unconfirmed.
"""

import os
import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import assistant
import llm

QUESTIONS = [
    "What's the base price of a 2-bed in Block B?",
    "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
    "An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the "
    "price after discounts, and what's due at booking?",
    "What's the transfer fee?",
    "What's the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output.md")

# Pause between questions. Google's free tier allows 5 requests per minute per model and
# a pricing question costs two (parameters, then the answer), so running six questions
# flat out trips the per-minute limit and spends the run in backoff. 13 seconds between
# questions keeps us under it. Set MGC_QUESTION_DELAY=0 if you are on a paid key.
QUESTION_DELAY_SECONDS = float(os.getenv("MGC_QUESTION_DELAY", "13"))


def main():
    print(f"Provider: {llm.describe()}")
    try:
        llm.api_key()
    except llm.MissingAPIKey as exc:
        print(f"\nCannot run: {exc}")
        return 1

    lines = [
        "# Part 1 - test output",
        "",
        "Six questions from the brief, run through `assistant.answer()`.",
        f"Provider/model: `{llm.describe()}`.",
        "",
        "All arithmetic below comes from `calculator.py` (deterministic Python), not from",
        "the model, and for a pricing question `calculator.summarise_quote()` writes the",
        "answer too - so every question costs exactly **one** model call, whose only job was",
        "to extract the parameters. Every answer carries the document and section it came",
        "from. The timing next to each status is the whole round trip; it is almost entirely",
        "the provider's own latency.",
        "",
    ]

    for n, question in enumerate(QUESTIONS, start=1):
        if n > 1 and QUESTION_DELAY_SECONDS:
            time.sleep(QUESTION_DELAY_SECONDS)
        print(f"\n{'=' * 78}\n[{n}/{len(QUESTIONS)}] {question}")
        lines += [f"## {n}. {question}", ""]
        try:
            r = assistant.answer(question)
        except Exception:
            # One failing question should not lose the other five.
            print(traceback.format_exc())
            lines += ["**FAILED**", "", "```", traceback.format_exc(), "```", ""]
            continue

        print(f"STATUS: {r['status']}  ({r['elapsed_seconds']}s, "
              f"{r['llm_calls']} model call{'' if r['llm_calls'] == 1 else 's'})")
        print(f"ANSWER: {r['answer']}")
        for s in r["sources"]:
            print(f"SOURCE: {s['document']} -> {s['section']}")
        for note in r["notes"]:
            print(f"NOTE:   {note}")

        lines += [f"**Status:** `{r['status']}` &nbsp;&middot;&nbsp; {r['elapsed_seconds']}s, "
                  f"{r['llm_calls']} model call{'' if r['llm_calls'] == 1 else 's'}",
                  "", r["answer"] or "_(empty)_", "", "**Sources:**", ""]
        if r["sources"]:
            for s in r["sources"]:
                lines.append(f"- {s['document']} -> *{s['section']}*")
        else:
            lines.append("- _(none)_")
        lines.append("")

        if r["calculation_text"]:
            print("\n" + r["calculation_text"])
            lines += ["**Calculation (Python, not the model):**", "", "```",
                      r["calculation_text"], "```", ""]
        if r["notes"]:
            lines += ["**Notes from the guardrails:**", ""]
            lines += [f"- {note}" for note in r["notes"]]
            lines.append("")

        lines += ["**Retrieved chunks (BM25):** "
                  + ", ".join(f"{h['section']} ({h['score']})" for h in r["retrieved"]), "", "---", ""]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
