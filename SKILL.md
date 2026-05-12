---
name: preop-clearance-triage
description: Use this skill when a clinician asks whether a patient needs pre-operative anesthesia clearance, what specialists are required, what risk factors are present, or to draft a pre-operative clearance request letter. Triggers on phrases like "does X need clearance," "pre-op assessment," "anesthesia clearance," "clearance for [procedure]," "draft a clearance letter," and "write a referral note." Requires FHIR patient context in the message metadata.
version: 1.0.0
tags:
  - preoperative
  - clearance
  - anesthesia
  - perioperative
  - risk-stratification
  - fhir
  - clinical-decision-support
inputModes:
  - text
outputModes:
  - text
---

# Pre-Operative Anesthesia Clearance Triage

This skill performs structured pre-operative anesthesia clearance assessment for a patient whose chart is accessible via FHIR. It is invoked by an A2A `SendMessage` call that carries patient FHIR context in `message.metadata` under the extension URI `https://app.promptopinion.ai/schemas/a2a/v1/fhir-context`.

## When to use this skill

- A clinician asks whether a patient needs pre-operative clearance for a procedure
- A clinician asks what specialists need to evaluate the patient before surgery
- A clinician asks for a risk assessment, RCRI score, or perioperative concerns
- A clinician asks for a clearance request letter, referral note, or pre-op clinical note
- A clinician asks a follow-up question about perioperative medication management (e.g., when to hold an anticoagulant)

## Inputs

| Field | Source | Required |
|---|---|---|
| Patient FHIR context | `message.metadata["https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"]` | Yes |
| Clinical query | `message.parts[].text` | Yes (free-text) |
| Session ID | `params.sessionId` | Optional (used as A2A `contextId`) |

The FHIR context object must include at minimum: `fhirUrl`, `fhirToken`, `patientId`. Optionally `fhirRefreshToken` and `fhirRefreshTokenUrl` for OAuth refresh support.

### Required FHIR scopes

`patient/Patient.rs`, `patient/Condition.rs`, `patient/MedicationRequest.rs`, `patient/Procedure.rs`, `patient/DocumentReference.rs`, `patient/Observation.rs`.

Optional: `patient/Encounter.rs`, `patient/AllergyIntolerance.rs`.

## What the skill does

1. **Fetches** the patient's FHIR R4 resources (Patient, Condition, MedicationRequest, Procedure, DocumentReference, Observation, Encounter, AllergyIntolerance) via the provided FHIR server URL and bearer token.
2. **Normalizes** the data into a `PatientSnapshot` with structured conditions, classified medications (drug class, specialty mapping), recent vitals, recent labs, parsed PCP and specialist notes.
3. **Evaluates** Tier-1 hard-stop triggers (active anticoagulation with specialty routing by indication, recent MI, decompensated CHF, oxygen-dependent state, severe uncontrolled HTN, seizure disorder, prior anesthesia complication, etc.) and Tier-2 risk factors.
4. **Computes** the Revised Cardiac Risk Index (RCRI) and an internal composite risk score.
5. **Detects** the requested procedure from the query (with latest-mention precedence for multi-turn conversations) and applies institutional-protocol escalation when the procedure is on the major-surgery list (cardiac surgery, neurosurgery, major vascular, major thoracic, major abdominal, organ transplant, major orthopedic).
6. **Decides** a disposition deterministically: `NO_CLEARANCE_NEEDED`, `CLEARANCE_RECOMMENDED`, `SPECIALIST_REQUIRED`, `ANESTHESIA_REVIEW_REQUIRED`, or `INSUFFICIENT_INFORMATION`.
7. **Enriches** the output with Claude Sonnet for plain-English clinical summary and 2–4 specific next steps, grounded in named guidelines (ACC/AHA, ASA, RCRI, STOP-BANG, ARISCAT, FDA drug labeling).
8. **Drafts clearance letters on request** — one to the PCP for general medical-clearance cases, or one focused letter per specialist when multiple consults are needed. Letters are generated in parallel and personalized with patient name, DOB, and the PCP's actual name extracted from FHIR `DocumentReference.author`.

## Outputs

The skill returns an A2A `Task` artifact with two parts:

- **`text`** (`mediaType: text/markdown`) — Human-readable assessment with disposition header, clinical summary, triggering factors, next steps, and any drafted clearance letters.
- **`data`** (`mediaType: application/json`) — Structured `ClearanceOutput` for programmatic consumption:

```json
{
  "disposition": "specialist_required",
  "risk_level": "high",
  "risk_score": 0,
  "rcri_score": 1,
  "confidence": 0.85,
  "recommended_specialties": ["cardiology"],
  "triggering_factors": ["Active anticoagulation therapy"],
  "active_medications": ["Rivaroxaban 20 mg", "Metoprolol succinate 50 mg", "..."],
  "clinical_summary": "Cardiology clearance is required before David can proceed...",
  "recommended_next_steps": ["..."],
  "specialist_findings": [{"specialty": "cardiology", "doctor_name": "Dr. Karen Chen", "..."}],
  "missing_information": ["Coagulation studies (PT/INR/aPTT)"],
  "clearance_letter": "Date: [Date]\n\nTo: Cardiology\n\nRE: Pre-operative Cardiology Clearance Request — David Okafor, DOB 1957-04-12\n\n..."
}
```

## Example invocations

Plain assessment:

> *"Does Sarah need clearance for her wisdom tooth extraction?"*

Procedure-anchored assessment (works mid-conversation, latest mention wins):

> *"What about clearance for open heart surgery?"*

Specialist-targeted letter:

> *"Draft a clearance letter for David's cardiologist regarding the rivaroxaban hold plan before his colonoscopy."*

Multi-specialist letter set (one letter per specialty, generated in parallel):

> *"Generate clearance letters for Linda's hip replacement."*

PCP-directed clinical note:

> *"Write a comprehensive clinical note for the PCP summarizing Sarah's pre-op assessment for brain surgery."*

Follow-up question (streaming, with web-search grounding):

> *"When should we hold his rivaroxaban?"*

## Clinical guidelines referenced

All reasoning is grounded in named, published standards. The skill does not invent thresholds, doctor names, or citations:

- ACC/AHA perioperative cardiovascular guidelines
- ASA preoperative evaluation framework
- RCRI (Revised Cardiac Risk Index)
- STOP-BANG (OSA screening)
- ARISCAT (postoperative pulmonary complication risk)
- FDA drug labeling for perioperative medication management

## Safety constraints

- The skill returns clinical decision support only. All output is for licensed clinician review and is **not** signed medical correspondence or medical advice.
- The skill does not persist any patient data. Each invocation is stateless and fetches FHIR resources fresh.
- FHIR scopes are explicitly declared in the agent card and auditable.
- Operating-team specialty consults (e.g., neurology consult for a neurosurgery patient) are never recommended — those teams are already involved in the procedure.
- When the disposition is `SPECIALIST_REQUIRED` or `ANESTHESIA_REVIEW_REQUIRED`, the clinical summary will state plainly that clearance is required *before* the procedure can proceed. Phrases like "medically cleared" are forbidden for these dispositions.

## Failure modes

- **Missing FHIR context** → `INSUFFICIENT_INFORMATION` disposition with a clear missing-data message
- **Expired FHIR token** → `INSUFFICIENT_INFORMATION` with a refresh prompt
- **Sparse patient chart** (no conditions, no meds, no PCP note) → `INSUFFICIENT_INFORMATION` with an enumerated list of missing fields
- **LLM enrichment failure** → falls back to a deterministic clinical summary generated from the engine output

## Implementation

Source: [github.com/jbrodev/clearpath](https://github.com/jbrodev/clearpath). License: MIT.

Endpoint: `POST /` (A2A v1 JSON-RPC). Discovery: `GET /.well-known/agent-card.json`.
