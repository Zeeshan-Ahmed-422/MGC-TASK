-- ============================================================================
-- Part 2 - the three queries. DuckDB dialect (standard SQL + ROW_NUMBER).
-- ============================================================================
--
-- These run against `leads_raw`, which is leads.csv loaded verbatim - messy city
-- spellings, duplicate rows and all. That is deliberate: the brief asks for the city
-- normalisation to happen inside the query, and running against the raw dump is the
-- only way to show it actually working.
--
-- run_sql.py splits this file on the "-- @query:" markers and runs each block.
-- ============================================================================


-- @query: 1. Conversion rate by lead source (200+ leads only, best first)
--
-- Two decisions worth noting:
--  * I deduplicate first. A lead entered twice by two agents would otherwise be
--    counted twice in its source's denominator, and if it converted, twice in the
--    numerator too. Against the real `leads` table this CTE is unnecessary, because
--    UNIQUE (crm_record_hash) makes duplicates impossible - see schema.sql.
--  * HAVING COUNT(*) >= 200 comes after grouping, which is the point of HAVING: a
--    source with 30 leads and 2 conversions shows 6.7% on nothing like enough data
--    to act on.
WITH deduped AS (
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY crm_record_hash
                   ORDER BY created_at, LENGTH(lead_id), lead_id
               ) AS rn
        FROM leads_raw
    )
    WHERE rn = 1
)
SELECT
    source,
    COUNT(*)                                        AS leads,
    SUM(converted)                                  AS conversions,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2)     AS conversion_rate_pct
FROM deduped
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;


-- @query: 2a. Find the duplicate leads and show which row wins
--
-- WHAT COUNTS AS A DUPLICATE
-- Not "same name" and not "same everything" - this export has no name, phone or email
-- column. It has crm_record_hash, the CRM's own hash of the lead's identity fields,
-- and 160 hashes appear on two rows each. Those pairs are byte-identical in every
-- other column including created_at, and their lead_ids are "MGC-104974" and
-- "MGC-104974-B": one human being, entered twice by two different agents. So the
-- duplicate key is crm_record_hash.
--
-- THE WINNER RULE (deterministic, which matters - the same input must always pick the
-- same survivor, or every downstream count moves around between runs):
--   1. earliest created_at  - the first agent to enter the lead owns it;
--   2. then shortest lead_id - breaks the tie when both rows share a timestamp, which
--      here they always do, and deterministically keeps "MGC-104974" over the
--      "-B" row the second agent created;
--   3. then lead_id alphabetically - a final tie-break so the ordering is total and
--      the result can never depend on scan order.
SELECT
    crm_record_hash,
    lead_id,
    created_at,
    source,
    converted,
    ROW_NUMBER() OVER (
        PARTITION BY crm_record_hash
        ORDER BY created_at, LENGTH(lead_id), lead_id
    ) AS rn,
    CASE WHEN ROW_NUMBER() OVER (
             PARTITION BY crm_record_hash
             ORDER BY created_at, LENGTH(lead_id), lead_id
         ) = 1 THEN 'KEEP' ELSE 'DROP (duplicate)' END AS verdict
FROM leads_raw
WHERE crm_record_hash IN (
    SELECT crm_record_hash FROM leads_raw GROUP BY crm_record_hash HAVING COUNT(*) > 1
)
ORDER BY crm_record_hash, rn
LIMIT 20;   -- 320 rows in 160 pairs; first 20 are enough to eyeball


-- @query: 2b. The canonical set - exactly one row per lead
--
-- Same window function, filtered to rn = 1. This is the row set everything downstream
-- should use, and it is what part3_ml/train.py reproduces in pandas so the model and
-- the SQL agree on what a lead is.
--
-- PREVENTING DUPLICATES AT SCHEMA LEVEL (the comment the brief asks for):
-- deduplicating in a query is a cure, not a prevention. The fix is UNIQUE
-- (crm_record_hash) on the leads table - see schema.sql, where the full reasoning and
-- the live-CRM version (UNIQUE on a normalised phone/email) are written out. With that
-- constraint in place the second agent's INSERT fails, and the correct insert path is
-- an UPSERT that enriches the existing lead instead of creating a second one.
-- run_sql.py demonstrates exactly that: loading the raw dump into `leads` is REJECTED
-- by the constraint until this dedup is applied.
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY crm_record_hash
               ORDER BY created_at, LENGTH(lead_id), lead_id
           ) AS rn
    FROM leads_raw
)
SELECT
    (SELECT COUNT(*) FROM leads_raw)                        AS rows_in_raw_dump,
    COUNT(*)                                                AS canonical_rows,
    (SELECT COUNT(*) FROM leads_raw) - COUNT(*)             AS duplicate_rows_removed,
    COUNT(DISTINCT crm_record_hash)                         AS distinct_leads
FROM ranked
WHERE rn = 1;


-- @query: 3. Conversion rate by city, normalising the messy spellings in the query
--
-- The dump has 21 spellings for 9 cities: 'Islamabad', 'ISLAMABAD', 'ISB',
-- 'Rawalpindi', 'RAWALPINDI', 'Rwp', 'Karachi', 'KARACHI', 'khi', and the UPPERCASE
-- variants of the rest. LOWER(TRIM(...)) collapses the case and whitespace variants;
-- the CASE handles the abbreviations, which no amount of case folding can fix.
--
-- The ELSE deliberately does NOT silently bucket an unknown value into 'Other': it
-- tags it 'UNMAPPED: <value>' so a new spelling appears in the report as an obvious
-- defect instead of quietly distorting a real city's rate. run_sql.py asserts that no
-- UNMAPPED rows come back.
WITH deduped AS (
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY crm_record_hash
                   ORDER BY created_at, LENGTH(lead_id), lead_id
               ) AS rn
        FROM leads_raw
    )
    WHERE rn = 1
),
normalised AS (
    SELECT
        CASE LOWER(TRIM(city))
            WHEN 'islamabad'  THEN 'Islamabad'
            WHEN 'isb'        THEN 'Islamabad'
            WHEN 'isl'        THEN 'Islamabad'
            WHEN 'rawalpindi' THEN 'Rawalpindi'
            WHEN 'rwp'        THEN 'Rawalpindi'
            WHEN 'pindi'      THEN 'Rawalpindi'
            WHEN 'karachi'    THEN 'Karachi'
            WHEN 'khi'        THEN 'Karachi'
            WHEN 'lahore'     THEN 'Lahore'
            WHEN 'lhr'        THEN 'Lahore'
            WHEN 'peshawar'   THEN 'Peshawar'
            WHEN 'pwr'        THEN 'Peshawar'
            WHEN 'faisalabad' THEN 'Faisalabad'
            WHEN 'fsd'        THEN 'Faisalabad'
            WHEN 'multan'     THEN 'Multan'
            WHEN 'gujranwala' THEN 'Gujranwala'
            WHEN 'grw'        THEN 'Gujranwala'
            WHEN 'abbottabad' THEN 'Abbottabad'
            WHEN 'atd'        THEN 'Abbottabad'
            ELSE 'UNMAPPED: ' || COALESCE(city, '(null)')
        END AS city,
        converted
    FROM deduped
)
SELECT
    city,
    COUNT(*)                                     AS leads,
    SUM(converted)                               AS conversions,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2)  AS conversion_rate_pct
FROM normalised
GROUP BY city
ORDER BY conversion_rate_pct DESC;
