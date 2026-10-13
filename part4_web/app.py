"""
Part 4 - one page that ties Part 1 and Part 3 together.

Usage (from the repo root):
    python part4_web/app.py          then open http://127.0.0.1:5000

Two forms on one page:
  * ask the document assistant a question -> answer, status, sources, and the
    calculation breakdown when the calculator was used;
  * enter a lead's details -> conversion probability and a high/medium/low band.

The dropdowns in the second form are built from the values the model was actually
trained on (they are saved inside model.pkl), so the page cannot offer a category the
model has never seen.

GRACEFUL DEGRADATION, which the brief asks for explicitly: if there is no API key, the
page still loads and the lead-scoring form still works. Only the question form shows an
error, and it says exactly which variable to set.
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

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model once at start-up. If it is missing we keep serving the
# page and tell the user to run train.py, rather than refusing to boot.
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=form_fields(), model_error=MODEL_ERROR,
                           provider=llm.describe())


@app.route("/favicon.ico")
def favicon():
    """Browsers request /favicon.ico on every page load, which was filling the log with
    404s. Redirect to the SVG in static/ instead."""
    return redirect(url_for("static", filename="favicon.svg"))


# GET as well as POST so the example-question links on the page work without any
# JavaScript - they are plain <a href="/ask?question=...">.
@app.route("/ask", methods=["GET", "POST"])
def ask():
    """Form 1 - the document assistant."""
    question = (request.values.get("question") or "").strip()
    result = None
    error = None

    if not question:
        error = "Please type a question."
    else:
        try:
            result = assistant.answer(question)
        except llm.MissingAPIKey as exc:
            # The whole point of catching this separately: no key is a configuration
            # problem with a clear fix, not a crash.
            error = str(exc)
        except llm.LLMError as exc:
            error = str(exc)
        except Exception:
            error = "Unexpected error:\n" + traceback.format_exc()

    return render_template("index.html", fields=form_fields(), model_error=MODEL_ERROR,
                           provider=llm.describe(), question=question,
                           answer_result=result, answer_error=error)


@app.route("/score", methods=["POST"])
def score():
    """Form 2 - lead scoring. Works with or without an API key."""
    if BUNDLE is None:
        return render_template("index.html", fields=form_fields(), model_error=MODEL_ERROR,
                               provider=llm.describe(), score_error=MODEL_ERROR)

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

    return render_template("index.html", fields=form_fields(), model_error=MODEL_ERROR,
                           provider=llm.describe(), score_result=result, score_error=error,
                           submitted=row)


if __name__ == "__main__":
    print(f"LLM provider: {llm.describe()}")
    try:
        llm.api_key()
    except llm.MissingAPIKey as exc:
        print(f"NOTE: {exc}")
        print("      The page will still load and lead scoring will still work.")
    if MODEL_ERROR:
        print(f"NOTE: {MODEL_ERROR}")
    # debug=False on purpose: the reloader would import this module twice and reload the
    # model and the BM25 index with it.
    app.run(host="127.0.0.1", port=5000, debug=False)
