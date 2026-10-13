# Part 1 - test output

Six questions from the brief, run through `assistant.answer()`.
Provider/model: `gemini / gemini-3.5-flash (falling back to gemini-3.1-flash-lite)`.

All arithmetic below comes from `calculator.py` (deterministic Python), not from
the model, and for a pricing question `calculator.summarise_quote()` writes the
answer too - so every question costs exactly **one** model call, whose only job was
to extract the parameters. Every answer carries the document and section it came
from. The timing next to each status is the whole round trip; it is almost entirely
the provider's own latency.

## 1. What's the base price of a 2-bed in Block B?

**Status:** `answered` &nbsp;&middot;&nbsp; 61.9s, 1 model call

2-Bed Standard in Block B (1,150 sq ft): base price PKR 22,425,000. Total price: PKR 22,425,000. Due at booking (10% token under the Standard Payment Plan (4 years)): PKR 2,242,500. Not included in that price: Utility connection and meter charges PKR 285,000, Parking slot (mandatory, 1 per unit) PKR 1,450,000, Maintenance advance (12 months) — 1,150 sq ft @ 9.5/sq ft/month PKR 131,100 (total PKR 1,866,100). Confirm before quoting a customer: No floor was given, so no floor premium was applied. Floors 13–19 add +4% and floors 20–22 add +7% (price list, Location Premiums).

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Base Prices (Block B)*
- MGC Aurora Heights — Price List & Payment Plan -> *Discounts*
- MGC Aurora Heights — Price List & Payment Plan -> *Standard Payment Plan (4 years)*

**Calculation (Python, not the model):**

```
2-Bed Standard, Block B
------------------------------------------------------------------------------
  Base price — 2-Bed Standard, Block B (1,150 sq ft)         PKR       22,425,000
      (straight lookup from the price list; source: MGC Aurora Heights — Price List & Payment Plan → Base Prices (Block B))
  Price after discounts (no discounts)                       PKR       22,425,000
      (22,425,000 − 0; source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Due at booking — token, 10.0% of the price after discounts PKR        2,242,500
      (10.0% x 22,425,000 (Standard Payment Plan (4 years)); source: MGC Aurora Heights — Price List & Payment Plan → Standard Payment Plan (4 years))
------------------------------------------------------------------------------
  PRICE AFTER DISCOUNTS: PKR 22,425,000
  DUE AT BOOKING:        PKR 2,242,500

  Payment schedule:
    Booking / token                 10.0%  PKR       2,242,500  Non-refundable after 30 days
    Confirmation                    10.0%  PKR       2,242,500  Due within 45 days of booking
    16 quarterly instalments        60.0%  PKR      13,455,000  3.75% each
    On possession                   20.0%  PKR       4,485,000  Due at handover

  Other charges (NOT included in the price above) — total 1,866,100:
    Utility connection and meter charges                                 285,000
    Parking slot (mandatory, 1 per unit)                               1,450,000
    Maintenance advance (12 months) — 1,150 sq ft @ 9.5/sq ft/month         131,100

  Confirm before quoting a customer:
    - No floor was given, so no floor premium was applied. Floors 13–19 add +4% and floors 20–22 add +7% (price list, Location Premiums).
```

**Notes from the guardrails:**

- Answer written by calculator.py from the price list, not by the model - the model only extracted the parameters ({"block": "B", "unit_type": "2-Bed Standard"}).

**Retrieved chunks (BM25):** Unit Configuration (10.654), Base Prices (Block B) (8.275), Base Prices (Block A) (7.15), Overview (5.999), Frequently Asked (3.745)

---

## 2. What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?

**Status:** `answered` &nbsp;&middot;&nbsp; 32.1s, 1 model call

2-Bed Standard in Block B (1,150 sq ft), floor 15, corner, Margalla-facing: base price PKR 22,425,000; plus 13% in location premiums (Floors 13–19 +4%, Corner unit +3%, Margalla-facing +6%) = PKR 2,915,250, giving a list price of PKR 25,340,250. Total price: PKR 25,340,250. Due at booking (10% token under the Standard Payment Plan (4 years)): PKR 2,534,025. Not included in that price: Utility connection and meter charges PKR 285,000, Parking slot (mandatory, 1 per unit) PKR 1,450,000, Maintenance advance (12 months) — 1,150 sq ft @ 9.5/sq ft/month PKR 131,100 (total PKR 1,866,100). IMPORTANT - the documents support two readings of this request, so here is the other one. 2-Bed Corner in Block B (1,310 sq ft), floor 15, Margalla-facing: base price PKR 26,855,000; plus 10% in location premiums (Floors 13–19 +4%, Margalla-facing +6%) = PKR 2,685,500, giving a list price of PKR 29,540,500. Total price: PKR 29,540,500. Due at booking (10% token under the Standard Payment Plan (4 years)): PKR 2,954,050. Not included in that price: Utility connection and meter charges PKR 285,000, Parking slot (mandatory, 1 per unit) PKR 1,450,000, Maintenance advance (12 months) — 1,310 sq ft @ 9.5/sq ft/month PKR 149,340 (total PKR 1,884,340). Confirm before quoting a customer: Priced as a 2-Bed Standard (1,150 sq ft) plus the +3% corner premium, which is exactly the worked example in the price list ('+4% +3% +6% = +13% over base'). Note Block B also sells a larger '2-Bed Corner' unit type (1,310 sq ft, base PKR 26,855,000). Confirm with the buyer which of the two they mean.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Base Prices (Block B)*
- MGC Aurora Heights — Project Brochure -> *Unit Configuration*
- MGC Aurora Heights — Price List & Payment Plan -> *Location Premiums*
- MGC Aurora Heights — Price List & Payment Plan -> *Discounts*
- MGC Aurora Heights — Price List & Payment Plan -> *Standard Payment Plan (4 years)*

**Calculation (Python, not the model):**

```
2-Bed Standard, Block B, floor 15, corner, Margalla-facing
------------------------------------------------------------------------------
  Base price — 2-Bed Standard, Block B (1,150 sq ft)         PKR       22,425,000
      (straight lookup from the price list; source: MGC Aurora Heights — Price List & Payment Plan → Base Prices (Block B))
  Margalla-facing orientation confirmed as possible          
      (Block B, floor 15, 2-Bed Standard; source: MGC Aurora Heights — Project Brochure → Unit Configuration)
  Premium: Floors 13–19 +4.0% of base                        PKR          897,000
      (4.0% x 22,425,000; source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  Premium: Corner unit +3.0% of base                         PKR          672,750
      (3.0% x 22,425,000; source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  Premium: Margalla-facing +6.0% of base                     PKR        1,345,500
      (6.0% x 22,425,000; source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  List price = base + premiums (+13% total)                  PKR       25,340,250
      (22,425,000 x (1 + 13%); source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  Price after discounts (no discounts)                       PKR       25,340,250
      (25,340,250 − 0; source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Due at booking — token, 10.0% of the price after discounts PKR        2,534,025
      (10.0% x 25,340,250 (Standard Payment Plan (4 years)); source: MGC Aurora Heights — Price List & Payment Plan → Standard Payment Plan (4 years))
------------------------------------------------------------------------------
  PRICE AFTER DISCOUNTS: PKR 25,340,250
  DUE AT BOOKING:        PKR 2,534,025

  Payment schedule:
    Booking / token                 10.0%  PKR       2,534,025  Non-refundable after 30 days
    Confirmation                    10.0%  PKR       2,534,025  Due within 45 days of booking
    16 quarterly instalments        60.0%  PKR      15,204,150  3.75% each
    On possession                   20.0%  PKR       5,068,050  Due at handover

  Other charges (NOT included in the price above) — total 1,866,100:
    Utility connection and meter charges                                 285,000
    Parking slot (mandatory, 1 per unit)                               1,450,000
    Maintenance advance (12 months) — 1,150 sq ft @ 9.5/sq ft/month         131,100

  Confirm before quoting a customer:
    - Priced as a 2-Bed Standard (1,150 sq ft) plus the +3% corner premium, which is exactly the worked example in the price list ('+4% +3% +6% = +13% over base'). Note Block B also sells a larger '2-Bed Corner' unit type (1,310 sq ft, base PKR 26,855,000). Confirm with the buyer which of the two they mean.
```

**Notes from the guardrails:**

- Answer written by calculator.py from the price list, not by the model - the model only extracted the parameters ({"unit_type": "2-Bed Standard", "corner": true, "margalla_facing": true, "floor": 15, "block": "B"}).

**Retrieved chunks (BM25):** Unit Configuration (17.406), Location Premiums (14.619), Base Prices (Block B) (8.091), Overview (7.146), Commercial Podium (5.424)

---

## 3. An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the price after discounts, and what's due at booking?

**Status:** `answered` &nbsp;&middot;&nbsp; 24.4s, 1 model call

3-Bed Executive in Block B (1,880 sq ft): base price PKR 39,480,000; less 8% of base in discounts (Overseas Pakistani (Roshan Digital Account) -2%, 50% upfront -6%) = PKR 3,158,400. Price after discounts: PKR 36,321,600. Due at booking (10% token under the Standard Payment Plan (4 years)): PKR 3,632,160. Taken literally, 50% upfront would be PKR 18,160,800. Not included in that price: Utility connection and meter charges PKR 285,000, Parking slot (mandatory, 1 per unit) PKR 1,450,000, Maintenance advance (12 months) — 1,880 sq ft @ 9.5/sq ft/month PKR 214,320 (total PKR 1,949,320). Confirm before quoting a customer: No floor was given, so no floor premium was applied. Floors 13–19 add +4% and floors 20–22 add +7% (price list, Location Premiums). The payment plan table gives percentages but does not say percent of what. We apply them to the price after discounts (what the buyer actually pays). If MGC intends them as percentages of the undiscounted list price, the token would be PKR 3,948,000 instead. Confirm with the marketing manager. This buyer is taking the '50% upfront' discount, but the documents publish only the standard 4-year plan — there is no separate schedule for a 50%-upfront buyer. Taken literally, 50% of the discounted price is PKR 18,160,800; the only documented booking/token figure is the standard plan's 10.0% = PKR 3,632,160. Confirm which applies with the marketing manager.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Base Prices (Block B)*
- MGC Aurora Heights — Price List & Payment Plan -> *Discounts*
- MGC Aurora Heights — Price List & Payment Plan -> *Standard Payment Plan (4 years)*

**Calculation (Python, not the model):**

```
3-Bed Executive, Block B
------------------------------------------------------------------------------
  Base price — 3-Bed Executive, Block B (1,880 sq ft)        PKR       39,480,000
      (straight lookup from the price list; source: MGC Aurora Heights — Price List & Payment Plan → Base Prices (Block B))
  Discount: Overseas Pakistani (Roshan Digital Account) −2.0% of base PKR         -789,600
      (2.0% x 39,480,000 (premiums are not discounted); source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Discount: 50% upfront −6.0% of base                        PKR       -2,368,800
      (6.0% x 39,480,000 (premiums are not discounted); source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Price after discounts (−8% of base)                        PKR       36,321,600
      (39,480,000 − 3,158,400; source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Due at booking — token, 10.0% of the price after discounts PKR        3,632,160
      (10.0% x 36,321,600 (Standard Payment Plan (4 years)); source: MGC Aurora Heights — Price List & Payment Plan → Standard Payment Plan (4 years))
------------------------------------------------------------------------------
  PRICE AFTER DISCOUNTS: PKR 36,321,600
  DUE AT BOOKING:        PKR 3,632,160

  Payment schedule:
    Booking / token                 10.0%  PKR       3,632,160  Non-refundable after 30 days
    Confirmation                    10.0%  PKR       3,632,160  Due within 45 days of booking
    16 quarterly instalments        60.0%  PKR      21,792,960  3.75% each
    On possession                   20.0%  PKR       7,264,320  Due at handover

  Other charges (NOT included in the price above) — total 1,949,320:
    Utility connection and meter charges                                 285,000
    Parking slot (mandatory, 1 per unit)                               1,450,000
    Maintenance advance (12 months) — 1,880 sq ft @ 9.5/sq ft/month         214,320

  Confirm before quoting a customer:
    - No floor was given, so no floor premium was applied. Floors 13–19 add +4% and floors 20–22 add +7% (price list, Location Premiums).
    - The payment plan table gives percentages but does not say percent of what. We apply them to the price after discounts (what the buyer actually pays). If MGC intends them as percentages of the undiscounted list price, the token would be PKR 3,948,000 instead. Confirm with the marketing manager.
    - This buyer is taking the '50% upfront' discount, but the documents publish only the standard 4-year plan — there is no separate schedule for a 50%-upfront buyer. Taken literally, 50% of the discounted price is PKR 18,160,800; the only documented booking/token figure is the standard plan's 10.0% = PKR 3,632,160. Confirm which applies with the marketing manager.
```

**Notes from the guardrails:**

- Answer written by calculator.py from the price list, not by the model - the model only extracted the parameters ({"unit_type": "3-Bed Executive", "discounts": ["overseas_pakistani", "fifty_percent_upfront"]}).

**Retrieved chunks (BM25):** Discounts (15.382), Overseas Buyers (11.463), Booking Process (8.357), Unit Configuration (7.771), Standard Payment Plan (4 years) (7.44)

---

## 4. What's the transfer fee?

**Status:** `conflict` &nbsp;&middot;&nbsp; 0.0s, 0 model calls

The documents disagree about the transfer fee: 2% of the current list price according to MGC Aurora Heights — Price List & Payment Plan (Other Charges); 2.5% of the current list price according to MGC Developments — Booking Policy & Sales FAQ (Transfers). Do not pick one. Quote both with their sources and confirm with the marketing manager before quoting a customer.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Other Charges*
- MGC Developments — Booking Policy & Sales FAQ -> *Transfers*

**Notes from the guardrails:**

- Answered without calling the model: this is a registered cross-document conflict, so the answer comes from the registry.
- The answer for the transfer fee is generated from the known-conflict registry, not written by the model: it is a registered cross-document disagreement, so both values and both sources are stated verbatim.

**Retrieved chunks (BM25):** Transfers (4.67), Other Charges (4.423), Notes (3.12), Frequently Asked (2.779), Overseas Buyers (2.72)

---

## 5. What's the rental yield on a 1-bed?

**Status:** `not_in_documents` &nbsp;&middot;&nbsp; 5.0s, 1 model call

MGC Developments does not publish rental yield projections, and sales staff are strictly prohibited from providing such projections verbally. Please direct this query to the marketing manager.

**Sources:**

- MGC Developments — Booking Policy & Sales FAQ -> *Frequently Asked*

**Retrieved chunks (BM25):** Frequently Asked (9.042), Unit Configuration (4.642), Notes (3.858), Base Prices (Block A) (3.116), Base Prices (Block B) (3.099)

---

## 6. Who is the anchor tenant?

**Status:** `unconfirmed` &nbsp;&middot;&nbsp; 11.4s, 1 model call

No anchor tenant has been confirmed for the commercial podium as of the date the project brochure was issued. Please consult the marketing manager for any updates.

**Sources:**

- MGC Aurora Heights — Project Brochure -> *Commercial Podium*

**Notes from the guardrails:**

- Re-labelled as unconfirmed: the answer itself says this is not confirmed in the documents.

**Retrieved chunks (BM25):** Commercial Podium (7.092), Frequently Asked (1.588), Cancellations and Refunds (1.579), Possession (1.517), Booking Process (1.513)

---
