# Part 1 - test output

Six questions from the brief, run through `assistant.answer()`.
Provider/model: `gemini / gemini-3.1-flash-lite`.

All arithmetic below comes from `calculator.py` (deterministic Python), not from
the model. Every answer carries the document and section it came from.

## 1. What's the base price of a 2-bed in Block B?

**Status:** `answered`

The base price for a 2-Bed Standard unit in Block B is PKR 22,425,000. Under the Standard Payment Plan, the amount due at booking is PKR 2,242,500. Please note that no floor was given, so no floor premium was applied. Floors 13–19 add +4% and floors 20–22 add +7% to the price. Additionally, utility connection charges (PKR 285,000), a mandatory parking slot (PKR 1,450,000), and 12 months of maintenance advance (PKR 131,100) are not included in the base price.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Base Prices (Block B)*

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

**Retrieved chunks (BM25):** Unit Configuration (10.654), Base Prices (Block B) (8.275), Base Prices (Block A) (7.15), Overview (5.999), Frequently Asked (3.745)

---

## 2. What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?

**Status:** `answered`

Please confirm with the customer which 2-bed unit they want. The '2-Bed Corner' (1,310 sq ft) total is PKR 29,540,500, with PKR 2,954,050 due at booking. Caveat: the 3% corner premium was not added as it is a base unit type; confirm if this should be added. The '2-Bed Standard' (1,150 sq ft) total is PKR 25,340,250, with PKR 2,534,025 due at booking. This includes all three premiums (13% total). Other charges (utility, parking, maintenance) apply to both.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Base Prices (Block B)*
- MGC Aurora Heights — Price List & Payment Plan -> *Location Premiums*
- MGC Aurora Heights — Project Brochure -> *Unit Configuration*

**Calculation (Python, not the model):**

```
2-Bed Corner, Block B, floor 15, Margalla-facing
------------------------------------------------------------------------------
  Base price — 2-Bed Corner, Block B (1,310 sq ft)           PKR       26,855,000
      (straight lookup from the price list; source: MGC Aurora Heights — Price List & Payment Plan → Base Prices (Block B))
  Margalla-facing orientation confirmed as possible          
      (Block B, floor 15, 2-Bed Corner; source: MGC Aurora Heights — Project Brochure → Unit Configuration)
  Premium: Floors 13–19 +4.0% of base                        PKR        1,074,200
      (4.0% x 26,855,000; source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  Premium: Margalla-facing +6.0% of base                     PKR        1,611,300
      (6.0% x 26,855,000; source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  List price = base + premiums (+10% total)                  PKR       29,540,500
      (26,855,000 x (1 + 10%); source: MGC Aurora Heights — Price List & Payment Plan → Location Premiums)
  Price after discounts (no discounts)                       PKR       29,540,500
      (29,540,500 − 0; source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Due at booking — token, 10.0% of the price after discounts PKR        2,954,050
      (10.0% x 29,540,500 (Standard Payment Plan (4 years)); source: MGC Aurora Heights — Price List & Payment Plan → Standard Payment Plan (4 years))
------------------------------------------------------------------------------
  PRICE AFTER DISCOUNTS: PKR 29,540,500
  DUE AT BOOKING:        PKR 2,954,050

  Payment schedule:
    Booking / token                 10.0%  PKR       2,954,050  Non-refundable after 30 days
    Confirmation                    10.0%  PKR       2,954,050  Due within 45 days of booking
    16 quarterly instalments        60.0%  PKR      17,724,300  3.75% each
    On possession                   20.0%  PKR       5,908,100  Due at handover

  Other charges (NOT included in the price above) — total 1,884,340:
    Utility connection and meter charges                                 285,000
    Parking slot (mandatory, 1 per unit)                               1,450,000
    Maintenance advance (12 months) — 1,310 sq ft @ 9.5/sq ft/month         149,340

  Confirm before quoting a customer:
    - '2-Bed Corner' is its own unit type with its own base price, so the +3% corner premium has NOT been added on top — that would charge for the corner twice. Confirm with the marketing manager if a 2-Bed Corner is also meant to carry the +3% premium.
```

**Retrieved chunks (BM25):** Unit Configuration (17.406), Location Premiums (14.619), Base Prices (Block B) (8.091), Overview (7.146), Commercial Podium (5.424)

---

## 3. An overseas Pakistani buyer pays 50% upfront on a 3-Bed Executive - what's the price after discounts, and what's due at booking?

**Status:** `answered`

The discounted price for the 3-Bed Executive is PKR 36,321,600. The standard booking amount is PKR 3,632,160. Caveats: No floor premium was applied as the floor is unspecified (floors 13–19 add +4%, 20–22 add +7%). The payment percentages are applied to the discounted price, but please confirm if these should be based on the original list price. Additionally, there is no separate payment schedule for a 50% upfront buyer, so confirm with the marketing manager whether the 10% booking figure remains applicable.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Discounts*
- MGC Aurora Heights — Price List & Payment Plan -> *Standard Payment Plan (4 years)*

**Calculation (Python, not the model):**

```
3-Bed Executive, Block B
------------------------------------------------------------------------------
  Base price — 3-Bed Executive, Block B (1,880 sq ft)        PKR       39,480,000
      (straight lookup from the price list; source: MGC Aurora Heights — Price List & Payment Plan → Base Prices (Block B))
  Discount: 50% upfront −6.0% of base                        PKR       -2,368,800
      (6.0% x 39,480,000 (premiums are not discounted); source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
  Discount: Overseas Pakistani (Roshan Digital Account) −2.0% of base PKR         -789,600
      (2.0% x 39,480,000 (premiums are not discounted); source: MGC Aurora Heights — Price List & Payment Plan → Discounts)
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

**Retrieved chunks (BM25):** Discounts (15.382), Overseas Buyers (11.463), Booking Process (8.357), Unit Configuration (7.771), Standard Payment Plan (4 years) (7.44)

---

## 4. What's the transfer fee?

**Status:** `conflict`

There is a conflict regarding the transfer fee. The 'MGC Aurora Heights — Price List & Payment Plan' (Section: Other Charges) states the fee is 2% of the current list price. However, the 'MGC Developments — Booking Policy & Sales FAQ' (Section: Transfers) states it is 2.5% of the current list price. Do not pick one; quote both figures with their respective sources and confirm with the marketing manager before quoting a customer.

**Sources:**

- MGC Aurora Heights — Price List & Payment Plan -> *Other Charges*
- MGC Developments — Booking Policy & Sales FAQ -> *Transfers*

**Retrieved chunks (BM25):** Transfers (4.67), Other Charges (4.423), Notes (3.12), Frequently Asked (2.779), Overseas Buyers (2.72)

---

## 5. What's the rental yield on a 1-bed?

**Status:** `not_in_documents`

MGC does not publish rental yield projections. Sales staff must not provide projections verbally. Please direct this query to the marketing manager.

**Sources:**

- MGC Developments — Booking Policy & Sales FAQ -> *Frequently Asked*

**Retrieved chunks (BM25):** Frequently Asked (9.042), Unit Configuration (4.642), Notes (3.858), Base Prices (Block A) (3.116), Base Prices (Block B) (3.099)

---

## 6. Who is the anchor tenant?

**Status:** `unconfirmed`

There is currently no anchor tenant confirmed for the commercial podium; anchor tenancy discussions are ongoing.

**Sources:**

- MGC Aurora Heights — Project Brochure -> *Commercial Podium*

**Notes from the guardrails:**

- Re-labelled as unconfirmed: the answer itself says this is not confirmed in the documents.

**Retrieved chunks (BM25):** Commercial Podium (7.092), Frequently Asked (1.588), Cancellations and Refunds (1.579), Possession (1.517), Booking Process (1.513)

---
