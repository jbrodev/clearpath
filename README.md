# ClearPath

**AI-powered pre-operative anesthesia clearance agent.**

ClearPath automates surgical pre-authorization by reasoning over live FHIR patient data. It combines deterministic clinical rule engines with Claude Sonnet to produce structured clearance decisions — complete with disposition, risk scores, triggering factors, and recommended next steps.

Built for [Agents Assemble — The Healthcare AI Endgame](https://agents-assemble.devpost.com/) on the [Prompt Opinion](https://www.promptopinion.ai/) platform.

---

## How it works

1. **FHIR ingestion** — fetches conditions, medications, vitals, labs, and clinical notes from an EHR via FHIR R4
2. **Clinical engines** — evaluates Tier 1 hard-stop triggers, Tier 2 risk factors, and RCRI cardiac risk score
3. **Claude Sonnet reasoning** — enriches the deterministic output with clinical narrative, specialist referral logic, and plain-language next steps
4. **A2A response** — returns a structured `ClearanceOutput` over the [A2A v0.3](https://google.github.io/A2A/) JSON-RPC protocol, propagating SHARP/FHIR context back so downstream agents inherit it

### Dispositions

| Disposition | Meaning |
|---|---|
| `CLEARED` | Patient may proceed to surgery |
| `CONDITIONAL_CLEARANCE` | Cleared with specific requirements |
| `SPECIALIST_REFERRAL` | Requires specialist evaluation before clearance |
| `DEFER` | Surgery should be postponed |
| `INSUFFICIENT_INFORMATION` | Cannot assess — data missing |

---

## Stack

- **Python 3.11+** / FastAPI
- **Claude Sonnet** (`claude-sonnet-4-6`) via Anthropic SDK
- **FHIR R4** patient data
- **A2A v0.3** agent protocol (A2A v1 agent-card spec)
- **SHARP / FHIR Context Extension** — `https://app.promptopinion.ai/schemas/a2a/v1/fhir-context`, with `offline_access` (refresh-token) support

### Declared FHIR scopes

ClearPath requests read access to: `Patient`, `Condition`, `MedicationRequest`, `Procedure`, `DocumentReference`, `Observation`, `Encounter`, `AllergyIntolerance`.

---

## Quickstart

```bash
git clone https://github.com/jbrodev/clearpath.git
cd clearpath
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
uvicorn clearpath.main:app --reload
```

Agent card available locally at:
```
GET http://localhost:8000/.well-known/agent-card.json
```

### Live deployment

ClearPath is deployed at:
```
https://clearpath-htiy.onrender.com/.well-known/agent-card.json
```

---

## Project structure

```
├── fhir/           # FHIR client, normalizer, note parser
├── engines/        # Trigger rules, medication classifier, decision builder
├── reasoning/      # Claude Sonnet enrichment engine + prompt templates
├── models/         # Clinical data models, A2A protocol models
├── data/           # Medication lists, trigger rules, synthetic patients
└── tests/          # E2E and unit tests
```

---

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required — Anthropic API key |
| `FHIR_BASE_URL` | FHIR server base URL |
| `FHIR_ACCESS_TOKEN` | Bearer token for FHIR auth |

---

## License

MIT
