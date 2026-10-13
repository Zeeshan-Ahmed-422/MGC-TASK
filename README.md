# MGC Developments - AI Developer build task

Four slices of one problem: the sales team wastes its day re-answering the same price
and policy questions out of PDFs, and it has far more leads than it can call. **Part 1**
is a document assistant that answers from MGC's brochure, price list and booking FAQ,
cites the document and section for every answer, refuses when the documents do not have
the answer, shows both values when two documents disagree, and does all arithmetic in
Python rather than in the language model. **Part 2** is a DuckDB schema for the CRM dump
plus three queries (conversion by source, duplicate detection with a deterministic
winner rule, conversion by city with the messy spellings normalised inside the query).
**Part 3** is a lead-scoring baseline trained on a time-based split with the seven
leakage columns identified and dropped; the top 100 scored leads in the newest 20% of
the data contain 29 conversions against 7.9 expected from 100 random calls. **Part 4**
is one bare HTML page that does both jobs.

---

## Setup

Needs Python 3.10 or newer. Nothing else - no database server, no Docker.

**Windows (PowerShell)**

```powershell
git clone <this-repo> mgc-task
cd mgc-task
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # paste your key into GOOGLE_API_KEY=
```

**macOS / Linux**

```bash
git clone <this-repo> mgc-task
cd mgc-task
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env             # paste your key into GOOGLE_API_KEY=
```

### The .env file

`.env` is gitignored. `.env.example` is the template. The default provider is Google AI
Studio because it has a free tier:

```ini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here        # https://aistudio.google.com/apikey
MODEL=gemini-2.5-flash
```

Switching provider is one line - `part1_assistant/llm.py` normalises all three:

```ini
LLM_PROVIDER=openai      MODEL=gpt-4o-mini        OPENAI_API_KEY=...
LLM_PROVIDER=anthropic   MODEL=claude-sonnet-5-5  ANTHROPIC_API_KEY=...
```

**`MODEL` is a fallback chain**, tried left to right, which is why the default has two
names in it. Google's free tier caps each model *separately* - 20 requests a day on
`gemini-3.5-flash` while I was building - and the models differ in speed by a lot: with
quota, `gemini-3.5-flash` answers in under 2 seconds; `gemini-3.1-flash-lite` takes about
10 but has its own allowance. So the chain is fast-one-first, dependable-one-second:
questions are quick while the daily allowance lasts and keep working after it runs out,
instead of the whole thing dying with a quota error. A single model name works fine too.

The code tells the failure modes apart: a short rate limit is retried with backoff; an
exhausted *daily* quota or a model this key cannot use (Google retired `gemini-2.5-flash`
for new keys mid-build) drops straight to the next model in the chain and says so;
anything else is raised. `run_test_questions.py` also waits 13 seconds between questions
to stay under the 5-per-minute limit - set `MGC_QUESTION_DELAY=0` on a paid key.

**Parts 2 and 3 need no API key at all.** Part 4 loads and scores leads without one; only
the question form needs it.

---

## Run it

Run all commands from the repo root. They are the same on Windows and macOS/Linux
(`python` vs `python3` is the only difference).

```bash
# Part 1 - the six questions from the brief, saved to part1_assistant/test_output.md
python part1_assistant/run_test_questions.py

# ask one question of your own
python part1_assistant/assistant.py "What's the transfer fee?"

# the calculator on its own, no API key needed - the arithmetic lives here
python part1_assistant/calculator.py

# the retriever on its own - shows the chunks and what BM25 matches
python part1_assistant/retrieval.py

# Part 2 - schema + three queries, saved to part2_sql/results.md
python part2_sql/run_sql.py

# Part 3 - trains the model, writes part3_ml/model.pkl and part3_ml/results.md
python part3_ml/train.py

# Part 4 - the web page (run Part 3 first, it needs model.pkl)
python part4_web/app.py      # then open http://127.0.0.1:5000
```

Generated files (`test_output.md`, `results.md`, `model.pkl`) are committed so you can
read the outputs without running anything.

---

## Part 1 - grounded document assistant

**Flow:** question -> BM25 retrieval over the three documents -> conflict check ->
LLM with one tool, seeing only the retrieved sections -> Python does any arithmetic ->
JSON answer -> deterministic guardrails.

| File | What it does |
|---|---|
| `retrieval.py` | Splits the three markdown documents into one chunk per `##` section, keeping document name + section name + text. BM25 search over ~22 chunks. |
| `pricing.json` | The price list hand-transcribed, with the document and section recorded next to every number. |
| `calculator.py` | All the arithmetic, in Python. Returns a step-by-step breakdown where each line carries its source. |
| `llm.py` | One `chat()` function over Gemini / OpenAI / Anthropic. |
| `assistant.py` | The prompt, the tool definition, and the guardrails. |
| `run_test_questions.py` | The six questions from the brief -> `test_output.md`. |

### Three routes, decided in Python before any call is made

Every question is routed before the model is involved, which is what keeps it to **one
model call or fewer**:

| Route | When | Model calls |
|---|---|---|
| **Registered conflict** | the question hits a `known_conflicts` topic | **0** - the answer is two documented values with their sources, built from the registry, so asking the model is pure latency |
| **Pricing** | the question mentions money *and* a unit | **1**, with the tool *forced* and a small prompt: the model's only job is reading the question into parameters, which needs the question, not the documents |
| **Everything else** | | **1**, with the retrieved sections and no tool schema at all |

The earlier version let the model decide and then corrected it: it would answer a pricing
question without the calculator, get caught by the guardrail, and be asked again with the
tool forced - two calls on exactly the questions people ask most, which is what made it
feel slow. If Python can already tell the question needs the calculator, the model should
never have been offered the choice. The forced-retry guardrail is still there as a
backstop, it just almost never has to fire now.

### Python writes the pricing answer too

A pricing question used to cost **two** sequential API calls: one for the model to extract
the parameters, a second for it to write the sentence around the numbers Python had
computed. That doubled the wait on exactly the questions people ask most, and the second
call was also where things went wrong - it dropped the figures, invented extra totals, and
mis-attributed values.

So Python now writes the pricing answer too (`calculator.summarise_quote()`). The model
does the one thing it is genuinely good at - turning "Margalla-facing corner on floor 15,
2-bed Block B" into parameters - and everything after that is deterministic. **Every
question now costs exactly one model call**, and the figures cannot be mangled between the
calculator and the screen. The page shows the elapsed time and the call count so you can
see it.

What is left is the provider's own latency: about 5 seconds a question on Google's free
tier, occasionally much worse when it is busy. The code does what it can about the spikes -
every request times out at 40 seconds and is retried rather than waited on (the free tier
sometimes takes 80 seconds for a call that normally takes 5), with short backoff. Raise it
with `MGC_LLM_TIMEOUT` if your key is slower; a paid key is the real fix.

### How the tool call actually works (and one provider bug worth knowing)

The textbook agent loop sends the tool's result back as a `role="tool"` message and lets
the model continue the same conversation. I built that first and Gemini rejected it:

```
400 Function call is missing a thought_signature in functionCall parts.
```

Gemini attaches an opaque signature to every function call and demands it back when you
replay that call in the history - and the OpenAI-compatible endpoint never exposes it, so
the replay can't be made valid. Rather than special-case one provider, the assistant makes
**two plain calls**: the first extracts the parameters, Python computes the answer, and the
second is a fresh call with the computed numbers written into the prompt as context. No
tool message is ever replayed, it behaves identically on all three providers, and the model
still only ever sees numbers Python produced. `llm.py` and `assistant.py` both carry the
reasoning in comments.

Two other things learned the hard way, both now handled: `MODEL` must be a model your key
can actually use (Google retired `gemini-2.5-flash` for new keys mid-build, and
`gemini-3.8-flash` / `gemini-flash-latest` were returning 503 "high demand" while
`gemini-3.5-flash` was fine), and the free tier throws transient 503s often enough that
one unlucky question would fail a six-question run - so `llm.py` retries transient errors
(429/5xx) with exponential backoff and logs each retry, and caps every request at 90
seconds so a hung call cannot block the run.

### Where the maths happens, and why

In `calculator.py`, never in the model. The model's only job on a pricing question is to
read "Margalla-facing corner unit on floor 15, 2-bed Block B" and fill in a form:
`{unit_type: "2-Bed Standard", block: "B", floor: 15, corner: true, margalla_facing: true}`.
Python does the rest and hands back every intermediate number with its source. Language
models are unreliable at multi-step percentage arithmetic, they fail silently when they
get it wrong, and a wrong price quoted to a customer costs a sale. Parameter extraction
is what they are good at; arithmetic is what a calculator is good at.

### How the premiums and discounts stack (read off the price list)

- **Premiums are percentages of the base price and they add.** The price list says so
  outright: *"Premiums are cumulative. A Margalla-facing corner unit on floor 15 carries
  +4% +3% +6% = +13% over base."* So list price = base x 1.13, not
  base x 1.04 x 1.03 x 1.06. That worked example is also the second test question, so
  the document hands you the expected answer.
- **Discounts are percentages of the base price only.** *"Full cash payment: 12% discount
  on base price. Location premiums are not discounted."* The calculator therefore
  computes the discount as `pct x base`, never `pct x (base + premiums)`.
- **Discounts add too.** The overseas discount is *"additional 2%, stackable with the
  above"*, so 50% upfront + overseas = 6% + 2% = 8% of base, not a 6% then 2%
  compounding.
- **Full cash and 50% upfront are mutually exclusive** - they are alternative ways of
  paying, so asking for both is rejected with an explanation rather than silently
  stacking to 18%.
- **Other charges stay outside the price.** Utility connection (285,000), the mandatory
  parking slot (1,450,000) and the maintenance advance are listed separately, as the
  price list lists them, so a quote is never silently inflated.

### Where I had to interpret, and what I did

Each of these appears as a `caveat` on the quote and is shown to the salesperson rather
than hidden:

1. **"10% at booking" - 10% of what?** The payment plan table gives shares but never says
   of which figure. I apply them to the price *after* discounts, because that is what the
   buyer actually pays. The caveat shows what the token would be under the other reading.
2. **"2-bed corner" is ambiguous, so the assistant answers twice.** Block B sells a
   distinct `2-Bed Corner` unit type (1,310 sq ft, base 26,855,000) *and* the price list
   has a separate +3% corner premium whose own worked example is a corner 2-bed on floor
   15. So "a Margalla-facing corner 2-bed on floor 15 in Block B" can mean:
   - a **2-Bed Standard** (1,150 sq ft) plus +4% +3% +6% = +13% -> **PKR 25,340,250**, or
   - the larger **2-Bed Corner** (1,310 sq ft) plus +4% +6% = +10% -> **PKR 29,540,500**.

   These are different units at different prices and the documents do not say which the
   customer means, so the calculator prices **both** and the prompt requires the model to
   show both and ask the salesperson to confirm the unit. Same philosophy as the transfer
   fee: where the documents don't settle it, don't pick. And if you name a `2-Bed Corner`
   explicitly, the +3% premium is *not* added on top - that would charge for the corner
   twice.
3. **The 50%-upfront buyer has no published schedule.** The documents publish only the
   4-year standard plan. The answer gives the documented 10% token *and* the literal 50%
   figure, and says the documents do not settle it.
4. **Margalla-facing is validated, not assumed.** The brochure says only Block B above
   floor 12 faces the Margallas, and never Studios or 1-Beds. Asking for a Margalla-facing
   studio is refused with the quote from the brochure instead of being priced.

### How grounding, conflicts and refusals work

- **Grounding.** The model only ever sees the retrieved sections, labelled with their
  document and section, and the prompt tells it it has no other knowledge of MGC Aurora
  Heights and must refuse rather than fill a gap.
- **Every answer is one of four statuses:** `answered`, `not_in_documents`, `conflict`,
  `unconfirmed`.
- **Conflicts.** `pricing.json` carries a `known_conflicts` registry - today, the transfer
  fee, which is **2%** in the price list (Other Charges) and **2.5%** in the booking FAQ
  (Transfers), each recorded with its own document, section and verbatim quote. If a
  question touches a registered topic, **the answer is generated from the registry in
  Python and the model's prose is discarded.** There is nothing for the model to add -
  the answer is "these two documents say these two different things" - and letting it
  write that sentence turned out to be a risk with no upside: in testing it produced the
  two correct values attributed to the *wrong* documents (2% to the FAQ, 2.5% to the price
  list, exactly backwards). Generating it from the registry means the value, the document
  and the section cannot come apart. The prompt also instructs the general case, so an
  unregistered disagreement can still be caught by the model - but only the registry is
  guaranteed.
- **Leaked context numbering.** The retrieved sections are labelled `[1]`-`[5]` so the
  model can refer to one, and it sometimes writes "document [2] states..." into the
  answer, which is meaningless to a salesperson. Since Python knows what each number was,
  it substitutes the real document name.
- **Refusals.** "What's the rental yield on a 1-bed?" returns `not_in_documents`. The
  interesting part is that the refusal is itself grounded: the FAQ explicitly says MGC
  does not publish rental yield projections and that staff must not give them verbally,
  so the answer cites that section while refusing.
- **Unconfirmed.** "Who is the anchor tenant?" returns `unconfirmed`, citing the
  brochure's Commercial Podium section, which says discussions are ongoing and no tenant
  has been confirmed.
- **Guardrails that do not trust the model.** Every one of these is code, not prompting,
  and each exists because a model actually got it wrong during testing:
  - An answer with no source is downgraded to `unconfirmed` (the brief: an answer without
    a source counts as wrong).
  - An unrecognised status becomes `unconfirmed`; a reply that is not valid JSON is passed
    through as raw text, flagged, and never presented as verified.
  - A registered conflict is forced, both sources added, and a single-value answer
    replaced.
  - **If the answer says something is not confirmed, the status is re-labelled
    `unconfirmed`** - for "who is the anchor tenant?" the model wrote the right sentence
    ("no anchor tenant has been confirmed") and still labelled it `answered`.
  - **If the answer is a refusal, the status is re-labelled `not_in_documents`** - the
    rental-yield answer came back word-perfect ("MGC does not publish rental yield
    projections... direct enquiries to the marketing manager") and labelled `answered`.
    The patterns are deliberately narrow, because "confirm with the marketing manager"
    also appears in pricing caveats and in the conflict answer, and neither of those is a
    refusal.
  - **If the calculator ran, its figures must appear in the answer.** The model once
    replied "refer to the calculated breakdown for the exact figures", which is useless to
    someone mid-quote, so Python checks its own numbers are present (digits only, so
    `36,321,600` and `36321600` both count) and appends them if not. Each fix is reported
    in the `notes` list and shown on the web page, so you can always see the guardrail
    fired rather than wondering.

---

## Part 2 - SQL (DuckDB)

`python part2_sql/run_sql.py` loads the CSV into an in-memory DuckDB, applies
`schema.sql`, **demonstrates that inserting the raw dump is rejected by the unique
constraint**, then loads the 9,000 canonical rows and runs the three queries. Output in
`part2_sql/results.md`.

### Schema reasoning (`schema.sql`)

- **One table.** The dump is one row per lead with no repeating groups - one source, one
  city, one budget, one outcome. There is nothing to normalise out without inventing
  structure the file does not have. The two things I *would* split in a real CRM are an
  `agents` table (the dump has `agent_experience_years` but no agent id, so the dimension
  cannot be built from this file) and an `interactions` table (`calls_made`,
  `total_call_seconds`, `whatsapp_replies` are aggregates of what should be one row per
  call, with a timestamp - that table is what you would need to score a lead at a point
  in time without leakage).
- **`lead_id` as primary key.** It is the CRM's own natural key, stable and already
  unique; a surrogate id would add nothing.
- **Types, not strings.** `TIMESTAMP` for `created_at`, `DECIMAL` for money (never binary
  floating point), `SMALLINT` for bedrooms, real `BOOLEAN`s for the flags.
- **`CHECK` constraints on the categoricals** - the nine real sources, the nine real
  cities, the six property types - so a typo like `facebook ads` cannot create a tenth
  source and quietly split a conversion report in two.
- **Nullable where the data is genuinely unknown.** `bedrooms` is null for 39% of rows and
  almost all of those are Plots and Commercial Shops, which have no bedrooms: that is
  structural, so `NOT NULL` with a 0 default would be a lie.
- **City stored twice:** `city_raw` exactly as the CRM sent it, so the row is still
  auditable against the source system, plus `city_normalized` with a `CHECK` that only
  allows the nine real cities, so a new unmapped spelling is rejected at write time.

### The dedup rule (query 2)

**What a duplicate is.** This export has no name, phone or email column. It has
`crm_record_hash` - the CRM's own hash of the lead's identity - and 160 hashes appear on
two rows each. Those 320 rows are byte-identical in every other column including
`created_at`, and their ids are `MGC-104974` and `MGC-104974-B`: one person, entered twice
by two agents. So the duplicate key is `crm_record_hash`.

**The winner rule,** which must be deterministic or every downstream count moves between
runs: `ROW_NUMBER() OVER (PARTITION BY crm_record_hash ORDER BY created_at, LENGTH(lead_id), lead_id)`
then keep `rn = 1`. Earliest `created_at` (the first agent to enter the lead owns it),
then shortest `lead_id` (which breaks the tie these rows always have and keeps
`MGC-104974` over the `-B` row), then `lead_id` alphabetically so the ordering is total.
9,160 rows -> 9,000.

**Preventing it at schema level:** `UNIQUE (crm_record_hash)`. A primary key on `lead_id`
does *not* help, because the second agent's row gets a different `lead_id` - the identity
of a lead is the person, so uniqueness has to be declared on the person. In a live CRM
with contact details that becomes `UNIQUE` on a normalised phone column (digits only, so
`+92 300 1234567` and `0300-1234567` collide) and on `lower(trim(email))`, with the insert
path as an UPSERT that enriches the existing lead instead of creating a second one.
`run_sql.py` proves the constraint works by trying the raw insert and showing the
rejection.

### City normalisation (query 3)

21 spellings for 9 cities. `LOWER(TRIM(city))` kills the case and whitespace variants;
a `CASE` maps the abbreviations (`isb`, `rwp`, `khi`) that case-folding cannot fix. The
`ELSE` deliberately does **not** bucket unknown values into 'Other' - it tags them
`UNMAPPED: <value>` so a new spelling shows up as an obvious defect instead of quietly
distorting a real city's rate, and `run_sql.py` asserts none come back.

### Results

Conversion by source (200+ leads): Referral 13.18%, Walk-in 10.28%, Facebook Ads 6.80%,
Google Search 6.50%, WhatsApp Campaign 6.33%, Property Portal 6.00%, Instagram 5.60%,
Billboard 4.33%, Expo Stall 2.95%. The two cheapest channels are the two best, by a
factor of four over the worst.

---

## Part 3 - lead scoring

`python part3_ml/train.py` -> `part3_ml/model.pkl`, `part3_ml/results.md`.

### Cleaning decisions

- **Deduplicated on `crm_record_hash`** using exactly the Part 2 winner rule, in pandas:
  9,160 -> 9,000 rows. The model and the SQL reports therefore agree on what a lead is.
- **Normalised `city`** with the same mapping as `queries.sql` (21 spellings -> 9 cities),
  and the script warns loudly if it meets a spelling the map does not cover.
- **Trimmed whitespace** on `source`, `property_type` and `area`. They are clean in this
  dump; a CRM export that is clean today will not be next month.
- **Parsed `created_at`** with `errors='coerce'` and dropped unparseable dates (none here)
  - a lead with no date cannot be placed on either side of a time split.
- **`area` missing for 477 leads (5%)**: kept, with missing encoded as its own `missing`
  category rather than imputed. "We never captured the area" is itself information about
  how the lead came in.
- **`bedrooms` missing for 3,602 leads (39%)**: median-imputed. Almost all are Plots,
  Commercial Shops and Farmhouses, which have no bedroom count - the missingness is
  structural and `property_type` carries the real signal.
- **`budget_pkr_lac` missing for 284 leads (3%)**: median-imputed.
- **Target**: `converted` as 0/1. 626 of 9,000 = **6.96% base rate**.
- Nothing in this file is constant or free text, so there was nothing to drop on those
  grounds.

### Dropped as leakage - the part that matters

These are recorded *while or after a lead is worked*, so at the moment we need a score
(a lead just arrived, nobody has called it) they are all 0 or unknown. Training on them
produces a model that tests brilliantly and cannot score a single new lead.

| Column | Why it is dropped |
|---|---|
| `token_amount_received_pkr` | Pure target leakage. No lead with a token of 0 ever converted - paying a token *is* booking. (719 leads have a token > 0 and only 634 are marked converted, so 85 paid and still fell through.) Known only after the sale. |
| `site_visits` | Post-contact. Conversion runs 4% at 0 visits to 67% at 4 visits, so it looks like a brilliant feature; every brand-new lead has 0. |
| `calls_made` | Post-contact. A lead we called six times is a lead we had already decided to chase - the model would learn our own past prioritisation, not the buyer's intent. |
| `total_call_seconds` | Post-contact, same reason. |
| `whatsapp_replies` | Post-contact engagement; zero for every lead at scoring time. |
| `first_response_minutes` | How fast *we* replied. An action we take, not an attribute of the lead, and it inverts cause and effect: leads the team liked got called back faster. |
| `agent_experience_years` | An attribute of the assigned agent, and assignment happens after creation. You cannot score an unassigned lead with it, and it would bake in which agents got the good leads. |

Also dropped: `lead_id` and `crm_record_hash` (identifiers - unique per row, no signal,
and a tree could memorise them), and `created_at` as a *feature* (it orders the split; as
a feature a raw timestamp only teaches the model "later rows look like this", which is
exactly the pattern that will not hold next month).

**Features kept** - all knowable the second a lead lands: `source`, `city`, `area`,
`property_type` (one-hot) and `budget_pkr_lac`, `bedrooms`, `is_overseas`,
`referred_by_existing_client`, `has_financing_approved` (median-imputed).

### Time-based split, and why it matters

Sorted by `created_at`, oldest 80% to train (7,200 leads, 2024-01-01 to 2025-05-27,
6.71% converted), newest 20% to test (1,800 leads, 2025-05-28 to 2025-10-01, 7.94%
converted).

In production this model scores leads that arrive *tomorrow* using leads that arrived
*before today*. A random split puts October's leads in training and June's in test, so
the model learns from the future and the test score flatters it - the evaluation answers
a question nobody asked. A time split answers the real one: trained on the past, does it
rank next month's leads? It is also the only split that exposes drift; note the base rate
itself moved from 6.71% to 7.94% between the two periods, which a random split would have
averaged away.

### Metric

**PR-AUC (average precision) is the headline**, with ROC-AUC alongside.

Only 7.94% of the test leads convert. Under imbalance like that, ROC-AUC is dominated by
how well the model orders the huge negative class, so it can look respectable while the
top of the list is wrong. PR-AUC only cares about precision and recall on the positives -
and a call list *is* precision on the positives: the team works down from the top, so
what matters is how many of the leads it flags are real.

| Metric | Baseline model | No skill |
|---|---|---|
| **PR-AUC** | **0.1914** | 0.0794 (the base rate) |
| ROC-AUC | 0.7133 | 0.5 (a coin flip) |

PR-AUC is 2.4x the no-skill line. Model: a scikit-learn `Pipeline` (one-hot + imputation +
`LogisticRegression(class_weight='balanced')`), no tuning. A `RandomForest` was fitted for
comparison and was slightly worse (PR-AUC 0.1767); the logistic regression is the
submitted model because it is simpler, and swapping to whichever model won on one test
set is how you start overfitting that test set.

### The number the sales team cares about

In the newest 1,800 leads, if they can only call 100:

- **Top 100 by score: 29 conversions.**
- 100 random leads: 7.9 expected (100 x the 7.94% base rate).
- **3.65x lift** - the same 100 calls, roughly 3.7 times the sales.

One caveat I would rather state than bury: `class_weight='balanced'` re-weights the
classes, so `predict_proba` returns numbers far above the real conversion rate (a good
lead scores ~0.83). They rank leads correctly, which is the job, but they are not
calibrated probabilities. That is why the web page shows a band and a percentile
("higher than 98% of leads we have seen") instead of claiming "an 83% chance of buying".

---

## Part 4 - the web interface

`python part4_web/app.py`, then:

- **http://127.0.0.1:5000/assistant** - page 1, ask the documents
- **http://127.0.0.1:5000/leads** - page 2, score a lead

(`/` redirects to page 1, and there is a nav link between them.) **No JavaScript at all**:
page 1 submits with GET so every answer has its own shareable, reloadable URL and the
example questions are ordinary links; page 2 submits with POST, because a lead's details
should not sit in a URL or a browser history.

**One module per page, and the server is threaded**, so the two tools are genuinely
independent: open them in two tabs and a slow question on page 1 does not block page 2 at
all. I measured it - page 2 answered in 0.05s while page 1 was still waiting on the model.
On one page, asking a question also re-rendered and wiped whatever the other module had on
screen, which is the other reason to split them.

The brief says there are no styling points, so there is no framework and no build step:
just one hand-written stylesheet (`part4_web/static/style.css`, ~200 lines of CSS grid and
custom properties) so the page is pleasant to demo rather than painful to look at. It
collapses to a single column on a phone. If you would rather see it truly bare, delete the
`<link rel="stylesheet">` line in `templates/base.html` - nothing else depends on it.

- **Page 1 (`/assistant`)** shows the answer, the status badge, the sources (document +
  section), the Python calculation breakdown when the calculator ran, any guardrail notes,
  which sections BM25 retrieved, and how long it took in model calls and seconds. The six
  questions from the brief are one-click links.
- **Page 2 (`/leads`)** scores a lead. The fields *are* the model's features, and the
  dropdown choices are read out of `model.pkl`, so the page cannot offer a category the
  model has never seen. Blank means "unknown" and is imputed by the pipeline. Output is a
  HIGH/MEDIUM/LOW band plus a percentile bar.
- Templates: `base.html` holds the layout and nav, `assistant.html` and `leads.html` fill
  in the content block.
- **No API key?** Both pages still load, page 1 shows exactly which variable to set and
  links to page 2, and page 2 keeps working.
- One note if you edit a template while the server is running: `debug=False`, so Flask
  does not reload templates - restart it. (`debug=True` would reload, but it also imports
  the module twice and reloads the model and BM25 index with it.)
- **If a second copy is already running, the app refuses to start.** On Windows a second
  Flask process can bind a port that is already in use rather than failing, and then
  requests go to whichever process the OS picks - usually the old one, running the code you
  just changed. It cost me three rounds of confusing debugging, so `app.py` probes the port
  first and tells you how to kill the stale process.

---

## Deploying it

The repo is ready to deploy as-is; the only thing that cannot live in it is the API key.

**Render** (free tier, what `render.yaml` is written for):

1. New → **Blueprint** → connect this repo. Render reads `render.yaml`, so the build and
   start commands and the Python version are already set.
2. It will ask for `GOOGLE_API_KEY` (it is marked `sync: false`, so it is never stored in
   the repo). Paste the key from https://aistudio.google.com/apikey.
3. Deploy. The live URL serves `/assistant` and `/leads` exactly as locally.

Anything that reads a `Procfile` (Railway, Heroku-style platforms) works too - same start
command, and set `GOOGLE_API_KEY`, `LLM_PROVIDER=gemini` and `MODEL` as environment
variables in that platform's dashboard.

Three decisions in there worth knowing:

- **The model is retrained at deploy time** (`buildCommand` runs `part3_ml/train.py`)
  rather than trusting the committed `model.pkl`. A pickle is tied to the scikit-learn
  version that wrote it, and the version the platform installs will not always match mine.
  Training takes a couple of seconds on 9,000 rows, so it is cheap insurance against a
  deploy that boots and then cannot score anything.
- **gunicorn, one worker, eight threads.** Flask's dev server is not meant to face the
  internet. One worker because each one loads its own copy of the model and the BM25
  index; threads because the work is waiting on an API, not CPU.
- **The free tier sleeps.** On Render's free plan the service spins down when idle, so the
  first request after a quiet spell takes ~30 seconds to wake it. That is the platform, not
  the app.

## Unfinished / what I'd do with more time

Honestly:

- **Conflict detection is only general for registered topics.** The transfer fee is in a
  hand-built registry in `pricing.json`, which guarantees it. The prompt asks the model to
  spot other disagreements, but nothing *guarantees* an unregistered one is caught. The
  real fix is a build step that extracts every labelled number from every document
  ("transfer fee: 2%") and diffs them across documents, flagging every mismatch into the
  registry automatically. That is what I would do next.
- **The documents are read fresh on every process start and the price list is transcribed
  by hand.** If the price list is revised, `pricing.json` has to be updated by hand, and
  nothing checks that it still agrees with the markdown. A test that re-parses the price
  list tables and asserts they match `pricing.json` would catch that drift, and is maybe
  twenty lines.
- **No tests.** The brief says test coverage is not being marked, so I spent the time on
  the hard cases instead. What I would write first: the two arithmetic cases against
  hand-computed numbers, the four guardrail paths (no source, bad status, non-JSON reply,
  forced conflict), and the Margalla validation refusals. I verified all of those by hand
  while building, with a stubbed LLM for the guardrails, but that scaffolding is not in
  the repo.
- **No calibration.** `CalibratedClassifierCV` on a held-out slice of the *training*
  period would turn the scores into probabilities a salesperson could reason about.
- **`has_financing_approved` deserves a second look.** It is one of the stronger features,
  and I have assumed it is captured at intake. If the CRM actually sets it after the first
  call, it is leakage too and the honest result is lower. I would check the CRM's audit log
  before trusting it - this is the first question I would ask on the call.
- **The interactions table.** The post-contact columns are not useless, they are just
  unusable *at intake*. With per-call timestamps you could build a second model that
  re-scores a lead as it is worked ("called twice, replied once, still no visit"), which is
  a more valuable product than a one-shot intake score. That needs data this export does
  not contain.
- **Part 4 is a single-process dev server** with no auth, no sessions, and the model loaded
  at import. Fine for a demo, not for the sales floor.
- **Retrieval is BM25 over 22 chunks.** Correct for three small documents. Thirty documents
  and I would add embeddings, and a reranker before I added anything cleverer than that.

## AI tools used

**Claude Code (Claude Opus) was used to build this**, which is what the ground rules
invite. How it was actually used:

- I read the brief and all three documents first and decided the architecture myself -
  in particular the decision that the LLM must not touch arithmetic, and the four-status
  grounding contract. Those were the decisions worth making and they came before any code.
- Claude wrote most of the code to that design, and wrote it with the comments in place,
  which is why the comments explain *why* rather than restate the line.
- Where it needed correcting: it is happy to write a plausible-looking number into a data
  file, so **I hand-transcribed `pricing.json` from the price list and hand-checked every
  figure in the six test answers against the documents**. It also initially wrote that
  `token_amount_received_pkr > 0` held for converted leads *and no others*, which the data
  contradicts - 85 leads paid a token and still did not convert. I checked it against the
  CSV and corrected it. That is the general lesson: it is fast and good at structure, and
  every factual claim it makes about *your data* needs checking against the data.
- It also wrote the textbook agent loop for the tool call, which Gemini rejected outright
  (the `thought_signature` error above), and defaulted to a model name Google had already
  retired for new keys. Both only showed up by actually running the thing against a real
  key - which is the argument for running your code rather than reading it.
- The genuinely useful part was the volume of explanatory comments and the reasoning in
  this README, which would have taken me far longer to write out by hand than the code did.
