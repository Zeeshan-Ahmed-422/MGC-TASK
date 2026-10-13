"""
Deterministic price calculator for MGC Aurora Heights.

WHY THIS FILE EXISTS
--------------------
Language models are unreliable at arithmetic, and a wrong price quoted to a customer
costs a sale. So the LLM in assistant.py is never allowed to do the maths. It only
*extracts the parameters* from the salesperson's question ("2-bed, Block B, floor 15,
corner, Margalla-facing") and calls `quote()` here. All the numbers come from
pricing.json, which was hand-transcribed from the price list, and every line of the
breakdown carries the document + section it came from.

HOW THE MATHS WORKS (read straight off the price list)
------------------------------------------------------
1. Base price is a lookup by (block, unit type).
2. Location premiums are percentages OF BASE and they ADD, not compound. The price list
   spells this out: "Premiums are cumulative. A Margalla-facing corner unit on floor 15
   carries +4% +3% +6% = +13% over base."
   -> list price = base * (1 + sum(premium_pcts)/100)
3. Discounts are also percentages OF BASE and they ADD ("additional 2%, stackable").
   The price list says "Location premiums are not discounted", so the discount is
   pct * base, never pct * (base + premiums).
   -> net price = list price - base * (sum(discount_pcts)/100)
4. The payment plan splits the net price. Booking / token = 10%.
5. "Other charges" (utilities, parking, maintenance) are listed separately and are
   never folded into the quoted price, because the price list keeps them separate.

Where the documents are genuinely ambiguous, the function returns a `caveats` list and
sets `needs_confirmation = True` instead of silently guessing. Those cases are:
  - the payment-plan percentages don't say "percent of what" (we use the net price);
  - a "2-bed corner" can mean the separate 2-Bed Corner unit type (its own bigger area
    and base price) or a 2-Bed Standard with the +3% corner premium;
  - the "50% upfront" discount has no payment schedule of its own in the documents.
"""

import json
import os
import sys

# pricing.json lives next to this file, so build an absolute path. Without this, the
# script would only work when run from inside part1_assistant/.
_HERE = os.path.dirname(os.path.abspath(__file__))
PRICING_PATH = os.path.join(_HERE, "pricing.json")

with open(PRICING_PATH, encoding="utf-8") as fh:
    PRICING = json.load(fh)

# Convenient shorthands for the document names used in the `source` of every step.
DOC_PRICE_LIST = PRICING["_documents"]["price_list"]
DOC_BROCHURE = PRICING["_documents"]["brochure"]

# The two mutually exclusive payment-style discounts: you either pay all cash or 50%
# upfront, not both.
_PAYMENT_STYLE_DISCOUNTS = ("full_cash", "fifty_percent_upfront")

TOP_FLOOR = 22  # 22-storey tower (brochure, Overview)


class CalculationError(ValueError):
    """Raised when the request contradicts the documents (e.g. a Margalla-facing studio).

    We raise instead of returning a number, because a quote built on an impossible
    unit is worse than no quote. assistant.py passes the message back to the LLM so it
    can tell the salesperson exactly what is wrong.
    """


def _source(section, document=DOC_PRICE_LIST):
    """Every number in the breakdown is tagged with where it came from."""
    return {"document": document, "section": section}


def _money(x):
    """Round to whole rupees. Percentages of round base prices are near-exact anyway;
    this just stops 25340249.999999996 appearing in a quote."""
    return round(x, 2)


def list_unit_types(block=None):
    """Unit types we have a base price for. Used for validation and by the web form."""
    if block:
        return [k for k in PRICING["base_prices"][block] if not k.startswith("_")]
    types = []
    for blk in ("A", "B"):
        for k in PRICING["base_prices"][blk]:
            if not k.startswith("_") and k not in types:
                types.append(k)
    return types


def base_price(unit_type, block):
    """Look up the base price for a unit type in a block.

    The two blocks have different price lists (Block B is more expensive per sq ft),
    and some unit types only exist in one block, so the block is required.
    """
    block = (block or "").strip().upper()
    if block not in ("A", "B"):
        raise CalculationError(
            f"Block must be 'A' or 'B' (got {block!r}). "
            "MGC Aurora Heights has two blocks (brochure, Overview)."
        )

    table = PRICING["base_prices"][block]
    if unit_type not in table:
        available = ", ".join(list_unit_types(block))
        raise CalculationError(
            f"There is no base price for a {unit_type!r} in Block {block}. "
            f"Block {block} price list covers: {available}."
        )
    return table[unit_type], _source(f"Base Prices (Block {block})")


def _floor_premium(floor):
    """Return (label, pct, source) for the floor band, or None for floors with no premium.

    Floors 1-12 carry no premium in the price list (floors 1-3 are the commercial
    podium; residential floors 4-12 are at base price).
    """
    if floor is None:
        return None
    if not isinstance(floor, int) or floor < 1 or floor > TOP_FLOOR:
        raise CalculationError(
            f"Floor must be a whole number between 1 and {TOP_FLOOR} "
            f"(got {floor!r}); the tower is {TOP_FLOOR} storeys (brochure, Overview)."
        )
    for band in PRICING["premiums"]["floor_bands"]:
        if band["min_floor"] <= floor <= band["max_floor"]:
            return band["label"], band["pct"], _source("Location Premiums")
    return None


def _validate_margalla(unit_type, block, floor):
    """The brochure limits which units can be Margalla-facing. Enforce it."""
    brochure_section = "Unit Configuration"
    if unit_type in ("Studio", "1-Bed Standard"):
        raise CalculationError(
            f"A {unit_type} cannot be Margalla-facing: the brochure says "
            "'Studio and 1-Bed units are not available with a Margalla-facing orientation'."
        )
    if block != "B":
        raise CalculationError(
            "Only Block B units face the Margalla range: the brochure says "
            "'All units above floor 12 in Block B face the Margalla range'. "
            f"This request is for Block {block}."
        )
    if floor is not None and floor <= 12:
        raise CalculationError(
            f"Floor {floor} is not Margalla-facing: the brochure says only units "
            "'above floor 12 in Block B' face the Margalla range."
        )
    return _source(brochure_section, DOC_BROCHURE)


def quote(
    unit_type,
    block,
    floor=None,
    corner=False,
    margalla_facing=False,
    discounts=(),
    plan="standard",
    additional_parking_slots=0,
):
    """Price a unit and return a full step-by-step breakdown.

    Args:
        unit_type: e.g. "2-Bed Standard", "3-Bed Executive" (see list_unit_types()).
        block: "A" or "B".
        floor: 1-22, or None if the salesperson didn't say (then no floor premium is
            applied and a caveat is added, rather than guessing a floor).
        corner: True for the +3% corner premium.
        margalla_facing: True for the +6% view premium.
        discounts: any of "full_cash", "fifty_percent_upfront", "overseas_pakistani",
            "second_unit".
        plan: payment plan key, currently only "standard" exists in the documents.
        additional_parking_slots: extra slots beyond the one mandatory slot.

    Returns:
        A dict with the numbers, a `steps` list (each step has a label, a value and the
        document+section it came from), `caveats`, and `needs_confirmation`.
    """
    block = (block or "").strip().upper()
    discounts = list(discounts or [])
    steps = []
    caveats = []

    # ---------------------------------------------------------------- base price
    base, base_src = base_price(unit_type, block)
    area = PRICING["unit_areas_sqft"].get(unit_type)
    steps.append({
        "label": f"Base price — {unit_type}, Block {block}"
                 + (f" ({area:,} sq ft)" if area else ""),
        "detail": "straight lookup from the price list",
        "value": _money(base),
        "source": base_src,
    })

    # "2-bed corner" is ambiguous in the documents, so say so rather than guess.
    # Block B has a distinct "2-Bed Corner" unit type (1,310 sq ft, PKR 26,855,000) AND
    # a separate +3% "Corner unit" premium. The price list's own worked example applies
    # the +3% to a corner unit, which is what we follow.
    if corner and unit_type == "2-Bed Standard" and "2-Bed Corner" in PRICING["base_prices"].get("B", {}):
        caveats.append(
            "Priced as a 2-Bed Standard (1,150 sq ft) plus the +3% corner premium, which is "
            "exactly the worked example in the price list ('+4% +3% +6% = +13% over base'). "
            "Note Block B also sells a larger '2-Bed Corner' unit type (1,310 sq ft, base "
            "PKR 26,855,000). Confirm with the buyer which of the two they mean."
        )
    if corner and unit_type == "2-Bed Corner":
        # The corner-ness is already in this unit type's own (higher) base price, so
        # adding +3% again would charge for it twice. Don't.
        corner = False
        caveats.append(
            "'2-Bed Corner' is its own unit type with its own base price, so the +3% corner "
            "premium has NOT been added on top — that would charge for the corner twice. "
            "Confirm with the marketing manager if a 2-Bed Corner is also meant to carry the "
            "+3% premium."
        )

    # ------------------------------------------------------------------ premiums
    premium_rows = []

    floor_band = _floor_premium(floor)
    if floor_band:
        label, pct, src = floor_band
        premium_rows.append({"label": label, "pct": pct, "source": src})
    elif floor is None:
        caveats.append(
            "No floor was given, so no floor premium was applied. Floors 13–19 add +4% "
            "and floors 20–22 add +7% (price list, Location Premiums)."
        )

    if corner:
        c = PRICING["premiums"]["corner"]
        premium_rows.append({"label": c["label"], "pct": c["pct"], "source": _source("Location Premiums")})

    if margalla_facing:
        margalla_src = _validate_margalla(unit_type, block, floor)  # raises if impossible
        m = PRICING["premiums"]["margalla_facing"]
        premium_rows.append({"label": m["label"], "pct": m["pct"], "source": _source("Location Premiums")})
        steps.append({
            "label": "Margalla-facing orientation confirmed as possible",
            "detail": f"Block {block}, floor {floor}, {unit_type}",
            "value": None,
            "source": margalla_src,
        })

    # Premiums are percentages of base and they ADD (price list: "Premiums are cumulative"
    # with the +4% +3% +6% = +13% worked example).
    premium_pct_total = sum(r["pct"] for r in premium_rows)
    for r in premium_rows:
        r["amount"] = _money(base * r["pct"] / 100)
        steps.append({
            "label": f"Premium: {r['label']} +{r['pct']}% of base",
            "detail": f"{r['pct']}% x {base:,.0f}",
            "value": r["amount"],
            "source": r["source"],
        })

    premium_amount = _money(base * premium_pct_total / 100)
    list_price = _money(base + premium_amount)
    if premium_rows:
        steps.append({
            "label": f"List price = base + premiums (+{premium_pct_total:g}% total)",
            "detail": f"{base:,.0f} x (1 + {premium_pct_total:g}%)",
            "value": list_price,
            "source": _source("Location Premiums"),
        })

    # ----------------------------------------------------------------- discounts
    style = [d for d in discounts if d in _PAYMENT_STYLE_DISCOUNTS]
    if len(style) > 1:
        raise CalculationError(
            "'Full cash payment' (12%) and '50% upfront' (6%) are alternative payment "
            "styles — a buyer can only have one of them, not both."
        )

    discount_rows = []
    for key in discounts:
        spec = PRICING["discounts"].get(key)
        if spec is None or key.startswith("_"):
            valid = [k for k in PRICING["discounts"] if not k.startswith("_")]
            raise CalculationError(f"Unknown discount {key!r}. The price list has: {', '.join(valid)}.")
        discount_rows.append({
            "label": spec["label"],
            "pct": spec["pct"],
            # Discounts are on BASE only: "Location premiums are not discounted."
            "amount": _money(base * spec["pct"] / 100),
            "source": _source("Discounts"),
        })
        steps.append({
            "label": f"Discount: {spec['label']} −{spec['pct']}% of base",
            "detail": f"{spec['pct']}% x {base:,.0f} (premiums are not discounted)",
            "value": -_money(base * spec["pct"] / 100),
            "source": _source("Discounts"),
        })

    discount_pct_total = sum(r["pct"] for r in discount_rows)
    discount_amount = _money(base * discount_pct_total / 100)
    net_price = _money(list_price - discount_amount)
    steps.append({
        "label": "Price after discounts"
                 + (f" (−{discount_pct_total:g}% of base)" if discount_rows else " (no discounts)"),
        "detail": f"{list_price:,.0f} − {discount_amount:,.0f}",
        "value": net_price,
        "source": _source("Discounts"),
    })

    # -------------------------------------------------------------- payment plan
    plan_spec = PRICING["payment_plans"].get(plan)
    if plan_spec is None:
        raise CalculationError(
            f"Unknown payment plan {plan!r}. The documents only publish: "
            f"{', '.join(k for k in PRICING['payment_plans'])}."
        )

    # The plan table gives shares as percentages but never says "percent of what".
    # We take them as shares of the price the buyer actually pays (after discounts).
    # Flagged as a caveat because it is an interpretation, not a quote from the document.
    schedule = []
    for m in plan_spec["milestones"]:
        schedule.append({
            "milestone": m["milestone"],
            "pct": m["pct"],
            "amount": _money(net_price * m["pct"] / 100),
            "notes": m.get("notes", ""),
            "source": _source(plan_spec["_source"]["section"]),
        })
    due_at_booking = schedule[0]["amount"]
    steps.append({
        "label": f"Due at booking — token, {schedule[0]['pct']}% of the price after discounts",
        "detail": f"{schedule[0]['pct']}% x {net_price:,.0f} ({plan_spec['label']})",
        "value": due_at_booking,
        "source": _source(plan_spec["_source"]["section"]),
    })
    if discount_rows:
        # Only worth flagging when there IS a discount; with no discount the two readings
        # (percent of list price vs percent of discounted price) give the same number.
        caveats.append(
            "The payment plan table gives percentages but does not say percent of what. We apply "
            "them to the price after discounts (what the buyer actually pays). If MGC intends them "
            "as percentages of the undiscounted list price, the token would be "
            f"PKR {_money(list_price * schedule[0]['pct'] / 100):,.0f} instead. Confirm with the "
            "marketing manager."
        )

    # The "50% upfront" discount names an amount the buyer pays up front, but the
    # documents publish no separate schedule for it - only the 4-year standard plan.
    upfront_amount = None
    if "fifty_percent_upfront" in discounts:
        upfront_amount = _money(net_price * 0.5)
        caveats.append(
            "This buyer is taking the '50% upfront' discount, but the documents publish only "
            "the standard 4-year plan — there is no separate schedule for a 50%-upfront "
            f"buyer. Taken literally, 50% of the discounted price is PKR {upfront_amount:,.0f}; "
            f"the only documented booking/token figure is the standard plan's "
            f"{schedule[0]['pct']}% = PKR {due_at_booking:,.0f}. Confirm which applies with the "
            "marketing manager."
        )

    # ------------------------------------------------------------ other charges
    oc = PRICING["other_charges"]
    other_charges = [
        {
            "label": oc["utility_connection_and_meter"]["label"],
            "amount": _money(oc["utility_connection_and_meter"]["amount"]),
            "source": _source("Other Charges"),
        },
        {
            "label": oc["parking_slot_mandatory"]["label"],
            "amount": _money(oc["parking_slot_mandatory"]["amount"]),
            "source": _source("Other Charges"),
        },
    ]
    if additional_parking_slots:
        other_charges.append({
            "label": f"{oc['parking_slot_additional']['label']} x {additional_parking_slots}",
            "amount": _money(oc["parking_slot_additional"]["amount"] * additional_parking_slots),
            "source": _source("Other Charges"),
        })
    if area:
        ma = oc["maintenance_advance"]
        other_charges.append({
            "label": f"{ma['label']} — {area:,} sq ft @ {ma['rate_per_sqft_per_month']}/sq ft/month",
            "amount": _money(area * ma["rate_per_sqft_per_month"] * ma["months"]),
            "source": _source("Other Charges"),
            "notes": ma["notes"],
        })
    other_charges_total = _money(sum(c["amount"] for c in other_charges))

    return {
        "unit_type": unit_type,
        "block": block,
        "area_sqft": area,
        "floor": floor,
        "corner": corner,
        "margalla_facing": margalla_facing,
        "currency": PRICING["currency"],

        "base_price": _money(base),
        "premiums": premium_rows,
        "premium_pct_total": premium_pct_total,
        "premium_amount": premium_amount,
        "list_price": list_price,

        "discounts": discount_rows,
        "discount_pct_total": discount_pct_total,
        "discount_amount": discount_amount,
        "net_price": net_price,

        "payment_plan": plan_spec["label"],
        "payment_schedule": schedule,
        "due_at_booking": due_at_booking,
        "upfront_50_percent": upfront_amount,

        # Kept separate from the price on purpose - the price list keeps them separate.
        "other_charges": other_charges,
        "other_charges_total": other_charges_total,

        "steps": steps,
        "caveats": caveats,
        "needs_confirmation": bool(caveats),
    }


def transfer_fee_conflict():
    """The transfer fee is the one place the documents flatly disagree (2% vs 2.5%).

    Returned straight from pricing.json so the assistant can show both values with both
    sources instead of picking a winner.
    """
    return PRICING["known_conflicts"][0]


def summarise_quote(q, alternative=None):
    """Write the salesperson-facing answer for a quote, in Python.

    WHY THIS IS NOT DONE BY THE LLM
    Originally the flow was: model extracts parameters -> Python computes -> model writes
    the sentence. That is two sequential API calls, so a pricing question took twice as
    long as any other, and the second call was the one that kept going wrong - it dropped
    the figures ("refer to the breakdown"), invented extra totals, and mis-attributed
    numbers. Everything the sentence needs is already in the quote, so Python writes it:
    one API call instead of two, and the numbers cannot be mangled on the way out.

    The model is still doing the job it is good at - reading "Margalla-facing corner on
    floor 15, 2-bed Block B" and turning it into parameters.
    """
    cur = q["currency"]
    bits = []

    features = []
    if q["floor"]:
        features.append(f"floor {q['floor']}")
    if q["corner"]:
        features.append("corner")
    if q["margalla_facing"]:
        features.append("Margalla-facing")
    unit = f"{q['unit_type']} in Block {q['block']}"
    if q["area_sqft"]:
        unit += f" ({q['area_sqft']:,} sq ft)"
    if features:
        unit += ", " + ", ".join(features)

    bits.append(f"{unit}: base price {cur} {q['base_price']:,.0f}")

    if q["premiums"]:
        premium_list = ", ".join(f"{p['label']} +{p['pct']:g}%" for p in q["premiums"])
        bits.append(
            f"plus {q['premium_pct_total']:g}% in location premiums ({premium_list}) "
            f"= {cur} {q['premium_amount']:,.0f}, giving a list price of "
            f"{cur} {q['list_price']:,.0f}"
        )

    if q["discounts"]:
        discount_list = ", ".join(f"{d['label']} -{d['pct']:g}%" for d in q["discounts"])
        bits.append(
            f"less {q['discount_pct_total']:g}% of base in discounts ({discount_list}) "
            f"= {cur} {q['discount_amount']:,.0f}"
        )

    sentence = "; ".join(bits) + "."

    price_label = "Price after discounts" if q["discounts"] else "Total price"
    lines = [
        sentence,
        f"{price_label}: {cur} {q['net_price']:,.0f}. "
        f"Due at booking ({q['payment_schedule'][0]['pct']:g}% token under the "
        f"{q['payment_plan']}): {cur} {q['due_at_booking']:,.0f}.",
    ]

    if q["upfront_50_percent"]:
        lines.append(
            f"Taken literally, 50% upfront would be {cur} {q['upfront_50_percent']:,.0f}."
        )

    lines.append(
        "Not included in that price: "
        + ", ".join(f"{c['label']} {cur} {c['amount']:,.0f}" for c in q["other_charges"])
        + f" (total {cur} {q['other_charges_total']:,.0f})."
    )

    if alternative is not None:
        # The ambiguous 2-bed corner case: give both units rather than choosing one.
        lines.append(
            "IMPORTANT - the documents support two readings of this request, so here is "
            f"the other one. {summarise_quote(alternative)}"
        )

    if q["caveats"]:
        lines.append("Confirm before quoting a customer: " + " ".join(q["caveats"]))

    return " ".join(lines)


def quote_sources(q):
    """The distinct document+section pairs behind a quote, for the answer's citations."""
    seen = []
    for step in q["steps"]:
        src = {"document": step["source"]["document"], "section": step["source"]["section"]}
        if src not in seen:
            seen.append(src)
    return seen


def format_breakdown(q):
    """Render a quote as plain text. Used by the test runner and the CLI."""
    lines = [f"{q['unit_type']}, Block {q['block']}"
             + (f", floor {q['floor']}" if q["floor"] else "")
             + (", corner" if q["corner"] else "")
             + (", Margalla-facing" if q["margalla_facing"] else "")]
    lines.append("-" * 78)
    for s in q["steps"]:
        value = "" if s["value"] is None else f"{q['currency']} {s['value']:>16,.0f}"
        lines.append(f"  {s['label']:<58} {value}")
        lines.append(f"      ({s['detail']}; source: {s['source']['document']} → {s['source']['section']})")
    lines.append("-" * 78)
    lines.append(f"  PRICE AFTER DISCOUNTS: {q['currency']} {q['net_price']:,.0f}")
    lines.append(f"  DUE AT BOOKING:        {q['currency']} {q['due_at_booking']:,.0f}")
    lines.append("")
    lines.append("  Payment schedule:")
    for m in q["payment_schedule"]:
        lines.append(f"    {m['milestone']:<30} {m['pct']:>5}%  {q['currency']} {m['amount']:>15,.0f}  {m['notes']}")
    lines.append("")
    lines.append(f"  Other charges (NOT included in the price above) — total {q['other_charges_total']:,.0f}:")
    for c in q["other_charges"]:
        lines.append(f"    {c['label']:<60} {c['amount']:>15,.0f}")
    if q["caveats"]:
        lines.append("")
        lines.append("  Confirm before quoting a customer:")
        for c in q["caveats"]:
            lines.append(f"    - {c}")
    return "\n".join(lines)


if __name__ == "__main__":
    # The document titles contain en-dashes, and a Windows console defaults to cp1252,
    # which cannot encode them. Force UTF-8 on stdout so the breakdown prints anywhere.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Quick manual check: the two arithmetic questions from the brief.
    print("Q: 2-bed Block B, floor 15, corner, Margalla-facing")
    print(format_breakdown(quote("2-Bed Standard", "B", floor=15, corner=True, margalla_facing=True)))
    print()
    print("Q: overseas buyer, 50% upfront, 3-Bed Executive")
    print(format_breakdown(quote("3-Bed Executive", "B",
                                 discounts=["fifty_percent_upfront", "overseas_pakistani"])))
