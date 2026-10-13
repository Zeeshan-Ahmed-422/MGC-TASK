"""
Part 4 - the web interface: two pages, one per module.

Usage (from the repo root):
    python part4_web/app.py          then open http://127.0.0.1:5000

  /assistant   page 1 - ask the document assistant (needs an API key)
  /leads       page 2 - score a lead (needs no API key)
  /            redirects to page 1

WHY TWO PAGES RATHER THAN ONE
They are independent tools used by different people at different moments. On one page,
asking a question re-rendered everything and wiped whatever the other module had on
screen. Separate pages mean a salesperson can keep the lead-scoring form open in one tab
and look a price up in another, and the server is threaded, so a slow question on page 1
does not block page 2 at all.

Page 1 submits with GET so every answer has its own URL: it can be bookmarked, shared or
reloaded, the back button works, and the example questions are ordinary links with no
JavaScript anywhere. Page 2 submits with POST, because a lead's details are not something
to leave sitting in a URL or a browser history.

GRACEFUL DEGRADATION, which the brief asks for explicitly: with no API key, both pages
still load, page 1 shows exactly which variable to set and links to page 2, and page 2
keeps working normally.
"""

import os
import sys
import traceback

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

# Part 1's modules live in a sibling directory, not an installed package, so put that
# directory on the import path. Simplest thing that works for a four-part task repo;
# a real project would make these proper packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "part1_assistant"))

import assistant  # noqa: E402  (import after sys.path is set up)
import llm        # noqa: E402

MODEL_PATH = os.path.join(_HERE, "..", "part3_ml", "model.pkl")

# The six questions from the brief, offered as one-click links on page 1.
EXAMPLE_QUESTIONS = [
    "What's the base price of a 2-bed in Block B?",
    "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
    "An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the "
    "price after discounts, and what's due at booking?",
    "What's the transfer fee?",
    "What's the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model once at start-up. If it is missing we keep serving the
# pages and tell the user to run train.py, rather than refusing to boot.
# ---------------------------------------------------------------------------
BUNDLE = None
MODEL_ERROR = None
try:
    BUNDLE = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL_ERROR = (f"Could not load part3_ml/model.pkl ({exc}). "
                   f"Run: python part3_ml/train.py")


def band(score):
    """Turn a score into high / medium / low.

    The cut points are the 1/3 and 2/3 quantiles of the scores on the TRAINING set,
    saved by train.py. Using the training distribution means "high" means "in the top
    third of leads we have seen", and it keeps the test set out of a product decision.
    """
    cuts = BUNDLE["band_cuts"]
    if score >= cuts["high_above"]:
        return "HIGH"
    if score >= cuts["medium_above"]:
        return "MEDIUM"
    return "LOW"


def percentile(score):
    """Where this score sits in the training-set score distribution, 0-100.

    This is the honest way to present the model's output. Because the model is fitted
    with class_weight='balanced', predict_proba returns values far above the ~7% base
    rate - a good lead scores about 0.8. Those values rank leads correctly, which is
    what a call list needs, but they are not calibrated probabilities, so the page
    reports the rank rather than claiming "an 83% chance of buying".
    """
    points = BUNDLE["score_percentiles"]  # 101 quantiles, 0th to 100th
    below = sum(1 for p in points if p <= score)
    return min(100, max(0, below - 1))


def form_fields():
    """Describe the lead form: exactly the features the model uses, nothing else."""
    if BUNDLE is None:
        return {"categorical": {}, "numeric": [], "binary": []}
    binary = BUNDLE["binary_features"]
    return {
        "categorical": BUNDLE["category_choices"],
        # Numeric fields minus the binary flags, which get checkboxes instead.
        "numeric": [c for c in BUNDLE["numeric_features"] if c not in binary],
        "binary": binary,
    }


@app.route("/")
def index():
    """No content of its own - page 1 is the front door."""
    return redirect(url_for("assistant_page"))


@app.route("/favicon.ico")
def favicon():
    """Browsers request /favicon.ico on every page load, which was filling the log with
    404s. Redirect to the SVG in static/ instead."""
    return redirect(url_for("static", filename="favicon.svg"))


@app.route("/assistant")
def assistant_page():
    """Page 1 - the document assistant. GET only: the question is in the URL."""
    question = (request.args.get("question") or "").strip()
    result = None
    error = None

    if question:
        try:
            result = assistant.answer(question)
        except llm.MissingAPIKey as exc:
            # Caught separately on purpose: no key is a configuration problem with a
            # clear fix, not a crash.
            error = str(exc)
        except llm.LLMError as exc:
            error = str(exc)
        except Exception:
            error = "Unexpected error:\n" + traceback.format_exc()

    return render_template("assistant.html", active="assistant", provider=llm.describe(),
                           examples=EXAMPLE_QUESTIONS, question=question,
                           answer_result=result, answer_error=error)


@app.route("/leads", methods=["GET", "POST"])
def leads_page():
    """Page 2 - lead scoring. Works with or without an API key."""
    if BUNDLE is None:
        return render_template("leads.html", active="leads", provider=llm.describe(),
                               fields=form_fields(), model_error=MODEL_ERROR)

    if request.method == "GET":
        return render_template("leads.html", active="leads", provider=llm.describe(),
                               fields=form_fields(), model_error=None)

    binary = BUNDLE["binary_features"]
    row = {}
    error = None

    # Categorical fields come from <select>s built from the training data, so they are
    # already valid values; an empty choice is passed through as a missing value, which
    # the pipeline's imputer handles.
    for col in BUNDLE["categorical_features"]:
        value = (request.form.get(col) or "").strip()
        row[col] = value or None

    # Checkboxes: present in the form data means 1.
    for col in binary:
        row[col] = 1 if request.form.get(col) else 0

    # Free numeric fields. Blank is allowed and becomes a missing value (the pipeline
    # median-imputes it) - better than forcing the salesperson to invent a budget.
    for col in BUNDLE["numeric_features"]:
        if col in binary:
            continue
        raw = (request.form.get(col) or "").strip()
        if raw == "":
            row[col] = None
            continue
        try:
            row[col] = float(raw)
        except ValueError:
            error = f"{col} must be a number (got {raw!r})."
            row[col] = None

    result = None
    if error is None:
        feature_order = BUNDLE["categorical_features"] + BUNDLE["numeric_features"]
        X = pd.DataFrame([row])[feature_order]
        try:
            model_score = float(BUNDLE["pipeline"].predict_proba(X)[0, 1])
            result = {
                "score": round(model_score, 3),
                "band": band(model_score),
                "percentile": percentile(model_score),
                "inputs": row,
                "base_rate": BUNDLE["metrics"]["test_base_rate"],
            }
        except Exception:
            error = "Scoring failed:\n" + traceback.format_exc()

    return render_template("leads.html", active="leads", provider=llm.describe(),
                           fields=form_fields(), model_error=None,
                           score_result=result, score_error=error, submitted=row)


def port_already_in_use(host="127.0.0.1", port=5000):
    """True if something is already listening there.

    Worth the eight lines: on Windows a second Flask process can bind a port that is
    already in use instead of failing, and then requests go to whichever process the OS
    feels like - usually the OLD one, still running the code you just changed. That cost
    me three confusing debugging rounds (a page that would not pick up new templates, an
    answer that used a model name I had already replaced). Better to refuse to start.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) == 0


if __name__ == "__main__":
    # Hosting platforms tell the app which port to listen on, and expect 0.0.0.0 rather
    # than 127.0.0.1. Locally neither variable is set, so it stays on 127.0.0.1:5000.
    # (In production gunicorn binds the socket itself and this block never runs - see
    # the Procfile - but honouring PORT means `python part4_web/app.py` works as a
    # fallback start command too.)
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")

    if port_already_in_use(port=port):
        print(f"ERROR: something is already listening on port {port} - most likely an "
              "older copy of this app.\n"
              "       Stop it first (Ctrl+C in its terminal), or on Windows:\n"
              f'         powershell "Get-NetTCPConnection -LocalPort {port} -State Listen | '
              'ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"\n'
              "       Starting anyway would serve stale code without telling you.")
        sys.exit(1)

    print(f"LLM provider: {llm.describe()}")
    try:
        llm.api_key()
    except llm.MissingAPIKey as exc:
        print(f"NOTE: {exc}")
        print("      Both pages still load and lead scoring still works.")
    if MODEL_ERROR:
        print(f"NOTE: {MODEL_ERROR}")
    shown_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"\n  page 1: http://{shown_host}:{port}/assistant   (ask the documents)")
    print(f"  page 2: http://{shown_host}:{port}/leads       (score a lead)\n")
    # threaded=True so a slow question on page 1 does not block page 2 - the two modules
    # are genuinely usable at the same time, in two tabs. (Flask's default is already
    # threaded; it is passed explicitly because the behaviour is the point.)
    # debug=False: the reloader would import this module twice and reload the model and
    # the BM25 index with it.
    app.run(host=host, port=port, debug=False, threaded=True)
