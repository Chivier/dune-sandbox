## Data Source

gdpr:
https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
hipaa:
https://www.hhs.gov/sites/default/files/ocr/privacy/hipaa/administrative/combined/hipaa-simplification-201303.pdf

---

## Dataset Formats

### Law (`law_converted.jsonl`)

Each line is a JSON record with the following structure:

```json
{
  "law": "<source> <article/section>: <title>\n<content>",
  "metadata": {
    "source": "GDPR | HIPAA",
    "article": "Article 1",     // GDPR only
    "title": "...",             // GDPR only
    "section": "160.101"        // HIPAA only
  }
}
```

**Files:** `law/GDPR.jsonl` (99 records), `law/HIPAA.jsonl` (144 records)

---

### User Requests (`req_converted.jsonl`)

Each line is a JSON record with the following structure:

```json
{
  "prompt": "<user request text>",
  "metadata": {
    "sender_role": "healthcare provider | employer | ...",
    "consent_obtained": true | false,
    "authorization_obtained": true | false,
    "policy": "POL_164.502_PERMIT_001"
  }
}
```

Policy label format: `POL_<section>_<PERMIT|FORBID|AMBIGUOUS>_<id>`

**File:** `req/req.jsonl` (200 records)
