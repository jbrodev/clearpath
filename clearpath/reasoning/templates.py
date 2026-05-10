"""
LLM prompt templates for ClearPath.
The LLM explains and summarizes — it does not make the decision.
The decision is already made by the deterministic engine.
"""


SYSTEM_PROMPT = """You are ClearPath. You help medical staff prepare a pre-op clearance summary that is also readable by the patient.

Ground every recommendation in recognized industry standards: ACC/AHA perioperative cardiovascular guidelines, ASA preoperative evaluation, RCRI for cardiac risk, STOP-BANG for OSA, ARISCAT for pulmonary risk, FDA drug labeling, and the most recent specialty-society guidelines. Do not invent thresholds or recommendations.

Write only:
1. A clinical_summary in plain English (1-3 short sentences). Lead with the bottom line. If a clinical term is necessary, define it in the same sentence (e.g., "atrial fibrillation, an irregular heartbeat").
2. 2-4 specific next steps, one short line each.

Rules for next steps:
- Base next steps STRICTLY on the Tier 1 triggers and Tier 2 factors listed in the prompt.
- Do NOT recommend specialist evaluation, clearance, or workup for any specialty not explicitly listed in the Tier 1 triggers.
- Do NOT add cardiac workup or cardiology referral unless a cardiac Tier 1 trigger is present.
- Do NOT add general anesthesia or surgical risk steps beyond what the flagged triggers require.
- If the procedure is low-risk (colonoscopy, endoscopy, cataract, minor skin procedure), calibrate accordingly.

Specialist scope:
- Cardiology owns cardiac drugs and AFib/ischemia management.
- Endocrinology owns diabetes and thyroid drugs.
- Pulmonology owns inhalers, oxygen, and OSA.
- The PCP is the default prescriber and manager for everything not explicitly owned by a specialist.
- Do NOT attribute the full medication list to a single specialist whose note happens to be in the chart.

You do NOT make the clearance decision: validated rules already did.
You do NOT practice medicine or give medical advice. Output is for licensed clinician review.
Avoid em dashes; use commas or colons.
Be brief. No preamble. No filler."""


def build_reasoning_prompt(
    disposition: str,
    risk_level: str,
    risk_score: int,
    rcri_score: int,
    triggering_factors: list[str],
    tier2_factors: list[str],
    patient_context: dict,
    specialist_findings: list[dict],
    missing_info: list[str],
    user_query: str,
) -> str:

    age = patient_context.get("age", "unknown")
    sex = patient_context.get("sex", "unknown")
    conditions = patient_context.get("conditions", [])
    medications = patient_context.get("medications", [])
    vitals = patient_context.get("vitals", {})
    pcp_summary = patient_context.get("pcp_note_excerpt", "")

    conditions_text = ", ".join(conditions[:8]) if conditions else "none documented"
    meds_text = ", ".join(medications[:8]) if medications else "none documented"

    vitals_text = ""
    if vitals.get("systolic_bp"):
        vitals_text += f"BP {vitals['systolic_bp']}/{vitals.get('diastolic_bp', '?')} mmHg"
    if vitals.get("heart_rate"):
        vitals_text += f", HR {vitals['heart_rate']}"
    if vitals.get("bmi"):
        vitals_text += f", BMI {vitals['bmi']:.1f}"
    if not vitals_text:
        vitals_text = "not available"

    triggers_text = "\n".join(f"- {t}" for t in triggering_factors) if triggering_factors else "None identified"
    tier2_text = "\n".join(f"- {t}" for t in tier2_factors) if tier2_factors else "None"

    specialist_text = ""
    if specialist_findings:
        specialist_text = "\n".join(
            f"- {sf.get('specialty', '').title()}: {sf.get('summary', '')} ({sf.get('status', '')})"
            for sf in specialist_findings
        )
    else:
        specialist_text = "No recent specialist notes"

    missing_text = "\n".join(f"- {m}" for m in missing_info) if missing_info else "None identified"

    pcp_excerpt = pcp_summary[:600].strip() if pcp_summary else "No PCP note available"

    return f"""The clinical rule engine has determined the disposition. Write only the patient-facing summary and next steps.

DISPOSITION: {disposition.replace('_', ' ').upper()}
RISK LEVEL: {risk_level.upper()}
RISK SCORE: {risk_score}/15
RCRI SCORE: {rcri_score}/6

PATIENT:
Age: {age} | Sex: {sex}
Active conditions: {conditions_text}
Active medications: {meds_text}
Recent vitals: {vitals_text}

TIER 1 TRIGGERS:
{triggers_text}

TIER 2 FACTORS:
{tier2_text}

RECENT SPECIALIST HISTORY:
{specialist_text}

MISSING INFORMATION:
{missing_text}

PCP NOTE EXCERPT:
{pcp_excerpt}

ORIGINAL CLINICAL QUERY:
{user_query}

---
Write ONLY these two fields. Do not restate the disposition or scores.

CLINICAL_SUMMARY: (1-3 short sentences in plain English. Lead with the bottom line: why this disposition. If you must use a clinical term, define it in the same sentence.)

NEXT_STEPS:
1. (most urgent action, one short line)
2. (next action, one short line)
3. (third if needed, else omit)
4. (fourth if needed, else omit)

Plain English. Brief. No preamble."""
