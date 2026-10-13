"""
Part 3 - baseline lead-scoring model.

Usage (from the repo root):
    python part3_ml/train.py

Writes part3_ml/model.pkl (used by the Part 4 web page) and part3_ml/results.md.

THE ONE DECISION THAT MATTERS HERE IS WHICH COLUMNS NOT TO USE.
leads.csv contains columns recorded while a lead is being worked - calls made, call
seconds, WhatsApp replies, site visits, response time, and the token amount received.
A model trained on those scores beautifully and is useless, because at the moment we
want a score (a lead just arrived, nobody has called it yet) every one of them is 0 or
unknown. token_amount_received_pkr is the extreme case: no lead with a token of 0 ever
converted, so the column is the outcome re-encoded. Training on it would produce a
near-perfect AUC and a model that cannot score a single new lead.

So the feature set is strictly what a CRM knows the second a lead lands: where it came
from, where the buyer is, what they want, their budget, and three intake flags.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "data", "leads.csv")
MODEL_PATH = os.path.join(_HERE, "model.pkl")
RESULTS_PATH = os.path.join(_HERE, "results.md")

TARGET = "converted"
TEST_FRACTION = 0.20  # newest 20% of leads by created_at
TOP_N = 100           # "if the team can only call 100 leads"

# ---------------------------------------------------------------------------
# Column decisions, written as data so they can be printed into results.md and
# defended one by one in the interview.
# ---------------------------------------------------------------------------

# Features known at the moment the lead is created - the only ones we may train on.
CATEGORICAL_FEATURES = ["source", "city", "area", "property_type"]
NUMERIC_FEATURES = [
    "budget_pkr_lac",
    "bedrooms",
    "is_overseas",
    "referred_by_existing_client",
    "has_financing_approved",
]

DROPPED_IDENTIFIERS = {
    "lead_id": "Identifier. Unique per row, so it carries no signal and would let a "
               "tree memorise individual leads.",
    "crm_record_hash": "Identifier (the CRM's hash of the buyer's contact details). "
                       "Used for deduplication, never as a feature.",
    "created_at": "Used to ORDER the train/test split, and dropped as a feature. A raw "
                  "timestamp only tells the model 'later rows look like this', which is "
                  "exactly the pattern that will not hold for next month's leads.",
}

DROPPED_LEAKAGE = {
    "token_amount_received_pkr": "Pure target leakage. Every one of the 634 converted "
                                 "leads in the raw file has a token > 0, and no lead with "
                                 "a token of 0 ever converted - paying a token IS "
                                 "booking. (719 leads have a token > 0, so 85 paid and "
                                 "were still not marked converted: bookings that fell "
                                 "through.) Either way the column is only known after the "
                                 "sale, so it cannot score a new lead.",
    "site_visits": "Only known after the lead is worked. Conversion runs 4% at 0 visits "
                   "to 67% at 4 visits, so it looks like a brilliant feature; a brand "
                   "new lead always has 0.",
    "calls_made": "Post-contact activity. A lead we have called six times is a lead we "
                  "already decided to chase - the model would learn our own past "
                  "prioritisation, not the buyer's intent.",
    "total_call_seconds": "Post-contact activity, same reason as calls_made.",
    "whatsapp_replies": "Post-contact engagement; zero for every lead at the moment we "
                        "need to score it.",
    "first_response_minutes": "How fast WE replied. Measured after the lead arrives, and "
                              "it is an action we take, not an attribute of the lead. "
                              "Including it would also mix up cause and effect: leads the "
                              "team liked got called back faster.",
    "agent_experience_years": "Attribute of the agent the lead was assigned to, not of "
                              "the lead, and assignment happens after creation. Scoring "
                              "an unassigned lead with it is impossible, and it would "
                              "bake in which agents got the good leads.",
}

# Not dropped for leakage, just worth knowing about.
DATA_QUALITY_NOTES = [
    "`area` is missing for 477 leads (5%). Kept, with the missing values encoded as "
    "their own 'missing' category rather than imputed - 'we never captured the area' is "
    "itself information about how the lead came in.",
    "`bedrooms` is missing for 3,602 leads (39%), and almost all of those are Plots, "
    "Commercial Shops and Farmhouses, which have no bedroom count. Missing here is "
    "structural, not a data-entry failure, so it is median-imputed and the property type "
    "carries the real signal.",
    "`budget_pkr_lac` is missing for 284 leads (3%); median-imputed.",
    "No column in this file is constant or free text, so there was nothing to drop on "
    "those grounds.",
]

# The messy city spellings, mapped exactly as part2_sql/queries.sql does, so the model
# and the SQL reports agree on what a city is.
CITY_MAP = {
    "isb": "Islamabad", "islamabad": "Islamabad",
    "rwp": "Rawalpindi", "rawalpindi": "Rawalpindi",
    "khi": "Karachi", "karachi": "Karachi",
    "lahore": "Lahore", "peshawar": "Peshawar", "faisalabad": "Faisalabad",
    "multan": "Multan", "gujranwala": "Gujranwala", "abbottabad": "Abbottabad",
}


def load_and_clean():
    """Load the CSV and apply the cleaning decisions. Returns (df, log_lines)."""
    log = []
    df = pd.read_csv(CSV_PATH)
    log.append(f"Loaded {len(df):,} rows, {df.shape[1]} columns from data/leads.csv.")

    # ---- dates -----------------------------------------------------------
    # errors='coerce' turns an unparseable date into NaT instead of raising, then we
    # drop those rows: a lead with no date cannot be placed on either side of a
    # time-based split. (In this file there are none, but the loader should not depend
    # on that.)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    bad_dates = df["created_at"].isna().sum()
    if bad_dates:
        df = df[df["created_at"].notna()]
        log.append(f"Dropped {bad_dates} rows with an unparseable created_at.")
    log.append(f"Date range: {df['created_at'].min():%Y-%m-%d} to {df['created_at'].max():%Y-%m-%d}.")

    # ---- dedupe, using exactly the Part 2 rule ----------------------------
    # Winner: earliest created_at, then shortest lead_id, then lead_id alphabetically.
    # sort_values + drop_duplicates(keep='first') is the pandas spelling of Part 2's
    # ROW_NUMBER() OVER (PARTITION BY crm_record_hash ORDER BY ...) = 1.
    before = len(df)
    df["_lead_id_len"] = df["lead_id"].str.len()
    df = (df.sort_values(["crm_record_hash", "created_at", "_lead_id_len", "lead_id"])
            .drop_duplicates("crm_record_hash", keep="first")
            .drop(columns="_lead_id_len"))
    log.append(f"Deduplicated on crm_record_hash with the Part 2 winner rule: "
               f"{before:,} -> {len(df):,} rows ({before - len(df)} duplicate rows removed).")

    # ---- normalise the categoricals ---------------------------------------
    unmapped = sorted(set(df["city"].str.strip().str.lower()) - set(CITY_MAP))
    if unmapped:
        # Loud, not silent: an unmapped spelling would otherwise become its own city.
        log.append(f"WARNING: unmapped city spellings found: {unmapped}")
    df["city"] = df["city"].str.strip().str.lower().map(CITY_MAP).fillna(df["city"])
    log.append(f"Normalised city: 21 raw spellings -> {df['city'].nunique()} cities "
               f"(same mapping as part2_sql/queries.sql).")

    for col in ["source", "property_type", "area"]:
        # Trim whitespace defensively. These three are already clean in this dump, but
        # a CRM export that is clean today will not be next month.
        # Done with map() rather than .astype("string").str.strip() on purpose: the
        # pandas "string" dtype uses pd.NA for missing values, and SimpleImputer raises
        # "boolean value of NA is ambiguous" on it. Plain object + np.nan is what
        # scikit-learn expects.
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else np.nan)

    # ---- target -----------------------------------------------------------
    df[TARGET] = df[TARGET].astype(int)
    log.append(f"Target `converted`: {df[TARGET].sum():,} of {len(df):,} leads = "
               f"{100 * df[TARGET].mean():.2f}% base rate.")

    return df, log


def time_split(df):
    """Oldest 80% train, newest 20% test, split on created_at.

    WHY NOT A RANDOM SPLIT. In production this model scores leads that arrive tomorrow
    using leads that arrived before today. A random split puts October's leads in the
    training set and June's in the test set, so the model gets to learn from the future
    and the test score flatters it. A time split asks the question we actually care
    about - "trained on the past, does it rank next month's leads?" - and it is the only
    split that exposes drift: if the market or the ad mix shifted, the time-split score
    drops and the random-split score does not.
    """
    df = df.sort_values("created_at").reset_index(drop=True)
    cut = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_pipeline(model):
    """One-hot + impute, then the classifier. No tuning: this is a baseline."""
    categorical = Pipeline([
        # Missing area becomes its own category, so "area not captured" is a signal the
        # model can use instead of being silently filled with the most common area.
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # handle_unknown='ignore' matters for the time split: a category that only
        # appears in the newer test data must not crash prediction.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("num", numeric, NUMERIC_FEATURES),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def evaluate(name, pipe, X_train, y_train, X_test, y_test):
    """Fit, score the test set, and return the metrics plus the top-100 business number."""
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    base_rate = float(y_test.mean())
    metrics = {
        "model": name,
        # PR-AUC (average precision) is the headline metric - see results.md for why.
        "pr_auc": float(average_precision_score(y_test, probs)),
        "pr_auc_no_skill": base_rate,
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "roc_auc_no_skill": 0.5,
        "test_rows": int(len(y_test)),
        "test_base_rate": base_rate,
    }

    # The number the sales team cares about: rank the test leads by score, take the top
    # 100, count how many actually converted, and compare with calling 100 at random.
    order = np.argsort(-probs)
    top = y_test.to_numpy()[order][:TOP_N]
    metrics["top_n"] = TOP_N
    metrics["top_n_conversions"] = int(top.sum())
    metrics["top_n_expected_random"] = round(TOP_N * base_rate, 1)
    metrics["top_n_lift"] = round(top.sum() / (TOP_N * base_rate), 2) if base_rate else None
    metrics["probs"] = probs
    return metrics


def main():
    df, clean_log = load_and_clean()
    train_df, test_df = time_split(df)

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X_train, y_train = train_df[feature_columns], train_df[TARGET]
    X_test, y_test = test_df[feature_columns], test_df[TARGET]

    split_note = (
        f"Train: {len(train_df):,} leads, {train_df['created_at'].min():%Y-%m-%d} to "
        f"{train_df['created_at'].max():%Y-%m-%d}, {100 * y_train.mean():.2f}% converted. "
        f"Test: {len(test_df):,} leads, {test_df['created_at'].min():%Y-%m-%d} to "
        f"{test_df['created_at'].max():%Y-%m-%d}, {100 * y_test.mean():.2f}% converted."
    )
    print("\n".join(clean_log))
    print(split_note)

    # class_weight='balanced' because only ~7% of leads convert: without it the model
    # minimises loss by predicting "no" for everything. It does not change the ranking
    # much, but it keeps the fitted coefficients interpretable.
    candidates = [
        ("LogisticRegression (baseline)",
         build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("RandomForest (comparison)",
         build_pipeline(RandomForestClassifier(
             n_estimators=300, min_samples_leaf=20, class_weight="balanced", random_state=42))),
    ]

    results = []
    for name, pipe in candidates:
        m = evaluate(name, pipe, X_train, y_train, X_test, y_test)
        m["pipeline"] = pipe
        results.append(m)
        print(f"\n{name}: PR-AUC {m['pr_auc']:.4f} (no-skill {m['pr_auc_no_skill']:.4f}), "
              f"ROC-AUC {m['roc_auc']:.4f}, "
              f"top {TOP_N}: {m['top_n_conversions']} conversions vs "
              f"{m['top_n_expected_random']} expected at random ({m['top_n_lift']}x)")

    # The logistic regression is the submitted baseline whatever the comparison says:
    # it is the simpler model, it is what the brief asks for, and swapping to whichever
    # model happened to win on one test set is how you start overfitting the test set.
    chosen = results[0]

    # Score bands for the web page, cut on the TRAIN distribution (using the test set to
    # choose thresholds would leak the test set into the product).
    train_probs = chosen["pipeline"].predict_proba(X_train)[:, 1]
    band_cuts = [float(np.quantile(train_probs, 2 / 3)), float(np.quantile(train_probs, 1 / 3))]

    # Dropdown choices for the web form, taken from the training data so the page can
    # only offer values the model has actually seen.
    category_choices = {
        col: sorted(x for x in train_df[col].dropna().unique()) for col in CATEGORICAL_FEATURES
    }

    bundle = {
        "pipeline": chosen["pipeline"],
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "category_choices": category_choices,
        # Binary intake flags get checkboxes in the UI rather than number fields.
        "binary_features": ["is_overseas", "referred_by_existing_client", "has_financing_approved"],
        "band_cuts": {"high_above": band_cuts[0], "medium_above": band_cuts[1]},
        "metrics": {k: v for k, v in chosen.items() if k not in ("pipeline", "probs")},
        "trained_on": split_note,
        "dropped_leakage": DROPPED_LEAKAGE,
        "dropped_identifiers": DROPPED_IDENTIFIERS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")

    # ---- results.md ------------------------------------------------------
    r = chosen
    lines = [
        "# Part 3 - lead scoring baseline, results", "",
        "Generated by `python part3_ml/train.py`.", "",
        "## Cleaning decisions", "",
    ]
    lines += [f"- {line}" for line in clean_log]
    lines += ["", "## Columns dropped as leakage", "",
              "Recorded only after a lead is worked, so unknowable at the moment we need a score:",
              ""]
    lines += [f"- **`{col}`** - {why}" for col, why in DROPPED_LEAKAGE.items()]
    lines += ["", "## Columns dropped as identifiers", ""]
    lines += [f"- **`{col}`** - {why}" for col, why in DROPPED_IDENTIFIERS.items()]
    lines += ["", "## Other data-quality notes", ""]
    lines += [f"- {n}" for n in DATA_QUALITY_NOTES]
    lines += ["", "## Features used", "",
              f"- Categorical (one-hot): {', '.join('`' + c + '`' for c in CATEGORICAL_FEATURES)}",
              f"- Numeric (median-imputed): {', '.join('`' + c + '`' for c in NUMERIC_FEATURES)}",
              "", "All of them are known the moment the lead lands in the CRM.", "",
              "## Time-based split", "", split_note, "",
              "Sorted by `created_at`, oldest 80% to train, newest 20% to test. A random split "
              "would let the model learn from leads that arrived *after* the ones it is tested "
              "on, which overstates performance and hides drift. This split answers the question "
              "we actually have: trained on the past, does it rank next month's leads?", "",
              "## Metric", "",
              "**PR-AUC (average precision) is the headline number.** Only "
              f"{100 * r['test_base_rate']:.1f}% of the test leads convert, and with imbalance "
              "like that ROC-AUC is dominated by how well the model orders the huge negative "
              "class - it looks respectable even when the top of the list is wrong. PR-AUC only "
              "cares about precision and recall on the positives, which is exactly what a call "
              "list is: a sales team works down from the top, so what matters is how many of the "
              "leads it flags are real. ROC-AUC is reported alongside because it is the number "
              "most people expect to see.", "",
              "| Metric | Model | No-skill baseline |", "|---|---|---|",
              f"| PR-AUC | **{r['pr_auc']:.4f}** | {r['pr_auc_no_skill']:.4f} (the base rate: "
              "guessing at random) |",
              f"| ROC-AUC | **{r['roc_auc']:.4f}** | 0.5000 (a coin flip) |", "",
              f"PR-AUC is {r['pr_auc'] / r['pr_auc_no_skill']:.1f}x the no-skill baseline.", "",
              "## What it means for the sales team", "",
              f"The test set is the newest {r['test_rows']:,} leads, of which "
              f"{100 * r['test_base_rate']:.2f}% converted.", "",
              f"- Call the **top {r['top_n']}** leads by score: **{r['top_n_conversions']} "
              f"conversions**.",
              f"- Call **{r['top_n']} leads at random**: **{r['top_n_expected_random']} "
              f"conversions** expected ({r['top_n']} x the base rate).",
              f"- That is a **{r['top_n_lift']}x lift** - the same 100 calls produce roughly "
              f"{r['top_n_conversions'] / r['top_n_expected_random']:.1f} times as many sales.",
              "",
              "## Model comparison", "",
              "| Model | PR-AUC | ROC-AUC | Top-100 conversions |", "|---|---|---|---|"]
    for m in results:
        lines.append(f"| {m['model']} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
                     f"{m['top_n_conversions']} |")
    lines += ["",
              "The submitted model is the logistic regression: it is the simpler of the two, it "
              "is a baseline as the brief asks, and picking whichever model won on a single test "
              "set is how you start overfitting that test set. No hyperparameter tuning was done "
              "at all.", "",
              "## Honest limitations", "",
              "- A ROC-AUC near 0.71 on intake-only features is a weak-but-real signal. It is "
              "enough to order a call list, which is all it is for; it is not enough to decide "
              "that a lead is hopeless.",
              "- The model's useful signal is concentrated in a few features (referral, "
              "financing approved, source, property type). It would be worth checking that those "
              "flags are actually populated at intake in the live CRM and not backfilled later - "
              "if `has_financing_approved` is often set after the first call, it is leakage too "
              "and I would need the CRM's audit log to tell.",
              "- No calibration. The probabilities are usable for ranking, which is the job; "
              "they should not be read as 'this lead has an 18% chance of buying' without a "
              "calibration curve.",
              ]
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {RESULTS_PATH}")

    # Print the headline numbers in a form that is easy to paste into the README.
    print("\n" + json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
