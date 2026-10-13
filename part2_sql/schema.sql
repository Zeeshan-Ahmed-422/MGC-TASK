-- ============================================================================
-- Part 2 - schema for the CRM lead dump (DuckDB dialect, standard SQL otherwise)
-- ============================================================================
--
-- WHY ONE TABLE
-- leads.csv is one row per lead with no repeating groups: a lead has exactly one
-- source, one city, one budget, one outcome. There is nothing to normalise into a
-- child table without inventing structure the CSV does not have. The two things I
-- WOULD split out in a real CRM are:
--   * agents  - the dump has agent_experience_years but no agent id, so an agent
--               dimension cannot be built from this file without guessing;
--   * contact history - calls_made / total_call_seconds / whatsapp_replies are
--               aggregates of what should be an interactions table (one row per
--               call or message, with a timestamp). That table is what you would
--               actually need to score a lead at a point in time without leakage.
-- Both are noted as "what I'd do with more time" rather than faked here.
--
-- WHY THE RAW + NORMALISED COLUMN PAIR FOR CITY
-- The dump has 21 spellings of 9 cities (Islamabad / ISLAMABAD / ISB, Rawalpindi /
-- RAWALPINDI / Rwp, khi, ...). I keep city_raw exactly as the CRM sent it (so the
-- record is still auditable against the source system) and store the cleaned value
-- next to it, so every query and the ML model use one consistent value. The CHECK on
-- city_normalized means a new, unmapped spelling is rejected at write time instead of
-- quietly creating a 22nd city.

DROP TABLE IF EXISTS leads;

CREATE TABLE leads (
    -- Natural key from the CRM ("MGC-104067"). It is stable and already unique in the
    -- dump, so no surrogate id is needed.
    lead_id                      TEXT        PRIMARY KEY,

    -- Every lead must have a creation time: it is what the time-based train/test split
    -- in Part 3 depends on, and a lead with no date cannot be ordered.
    created_at                   TIMESTAMP   NOT NULL,

    -- Lead source. NOT NULL and constrained to the nine values the business actually
    -- uses, so a typo ("facebook ads") cannot create a tenth source and silently split
    -- the conversion-rate report in two.
    source                       TEXT        NOT NULL
        CHECK (source IN ('Facebook Ads','Property Portal','Google Search','Instagram',
                          'Referral','Walk-in','WhatsApp Campaign','Expo Stall','Billboard')),

    city_raw                     TEXT        NOT NULL,
    city_normalized              TEXT        NOT NULL
        CHECK (city_normalized IN ('Islamabad','Rawalpindi','Lahore','Karachi','Peshawar',
                                   'Faisalabad','Multan','Gujranwala','Abbottabad')),

    -- Nullable: 477 leads have no area recorded. A nullable column is honest about
    -- "we never captured it"; an 'Unknown' string would pretend we did.
    area                         TEXT,

    property_type                TEXT        NOT NULL
        CHECK (property_type IN ('Apartment','Plot','Villa','Commercial Shop',
                                 'Penthouse','Farmhouse')),

    -- Budget in lac PKR (1 lac = 100,000). DECIMAL, not FLOAT: money should not be
    -- stored in binary floating point. Nullable because 284 leads never stated one,
    -- but if present it has to be positive.
    budget_pkr_lac               DECIMAL(12,2)
        CHECK (budget_pkr_lac IS NULL OR budget_pkr_lac > 0),

    -- Nullable by design, not by accident: a Plot or a Commercial Shop has no bedroom
    -- count. 3,602 of 9,160 rows are null and almost all are those property types.
    bedrooms                     SMALLINT
        CHECK (bedrooms IS NULL OR bedrooms BETWEEN 0 AND 10),

    -- ---- Columns below here are recorded AFTER the lead starts being worked. -------
    -- They are legitimate CRM data and worth storing, but Part 3 must not train on
    -- them (see README: leakage). Keeping them in the same table is fine; knowing
    -- which ones they are is the point.
    first_response_minutes       INTEGER     CHECK (first_response_minutes IS NULL OR first_response_minutes >= 0),
    calls_made                   INTEGER     NOT NULL DEFAULT 0 CHECK (calls_made >= 0),
    total_call_seconds           INTEGER     NOT NULL DEFAULT 0 CHECK (total_call_seconds >= 0),
    whatsapp_replies             INTEGER     NOT NULL DEFAULT 0 CHECK (whatsapp_replies >= 0),
    site_visits                  INTEGER     NOT NULL DEFAULT 0 CHECK (site_visits >= 0),
    agent_experience_years       DECIMAL(4,1) CHECK (agent_experience_years IS NULL OR agent_experience_years >= 0),

    -- Buyer attributes known at intake. Stored as BOOLEAN rather than 0/1 integers so
    -- the type says what the column means.
    is_overseas                  BOOLEAN     NOT NULL,
    referred_by_existing_client  BOOLEAN     NOT NULL,
    has_financing_approved       BOOLEAN     NOT NULL,

    -- Money received. In this dump token_amount_received_pkr > 0 holds for every
    -- converted lead and no unconverted one, i.e. it IS the outcome - the clearest
    -- leakage column in the file.
    token_amount_received_pkr    DECIMAL(14,2) NOT NULL DEFAULT 0
        CHECK (token_amount_received_pkr >= 0),

    -- The CRM's own hash of the lead's identity fields (name/phone/email are not in
    -- this export). Two rows with the same hash are the same human being entered
    -- twice. See the UNIQUE constraint below.
    crm_record_hash              BIGINT      NOT NULL,

    -- The target. BOOLEAN keeps it two-valued by type; the CHECK is belt and braces
    -- for loaders that would push an integer in.
    converted                    BOOLEAN     NOT NULL,

    -- ---- DUPLICATE PREVENTION AT SCHEMA LEVEL ---------------------------------
    -- This is the constraint that answers query 2. The dump contains 160 pairs of
    -- rows that share a crm_record_hash: the same lead entered twice by two agents,
    -- with lead_ids like MGC-104974 and MGC-104974-B. A PRIMARY KEY on lead_id does
    -- NOT stop that, because the second agent's row gets a different lead_id. The
    -- identity of a lead is the person, so the uniqueness has to be declared on the
    -- person, not on the row id.
    --
    -- In a live CRM where phone and email are present, this would instead be a
    -- UNIQUE on a normalised contact column, e.g.
    --     phone_normalized TEXT GENERATED ALWAYS AS (regexp_replace(phone,'\D','','g')),
    --     UNIQUE (phone_normalized)
    -- plus a UNIQUE (lower(trim(email))), so '+92 300 1234567', '03001234567' and
    -- '0300-1234567' all collide on insert. The insert path would then be an UPSERT
    -- that enriches the existing lead instead of creating a second one.
    UNIQUE (crm_record_hash),

    -- A lead cannot be created in the future, and this project's history starts in
    -- 2024. Cheap sanity check that catches clock and timezone bugs on load.
    CHECK (created_at >= TIMESTAMP '2024-01-01' AND created_at <= now())
);

-- Indexes for the access patterns the sales team actually has: "today's new leads",
-- "how is this source doing", "leads in this city".
CREATE INDEX idx_leads_created_at ON leads (created_at);
CREATE INDEX idx_leads_source     ON leads (source);
CREATE INDEX idx_leads_city       ON leads (city_normalized);
