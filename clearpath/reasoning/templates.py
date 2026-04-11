"""
LLM prompt templates for ClearPath.
The LLM explains and summarizes — it does not make the decision.
The decision is already made by the deterministic engine.
"""


SYSTEM_PROMPT = """You are ClearPath, a clinical decision support tool that helps medical staff prepare pre-operative clearance documentation.

Your role is ONLY to:
1. Write a concise clinical summary (2-4 sentences) of the patient's relevant clinical picture
2. Write 2-5 specific, actionable next steps appropriate to the disposition

You do NOT make the clearance decision — that has already been determined by validated clinical rules.
You do NOT practice medicine or give medical advice.
All output is for review by a licensed clinician before any action is taken.
Always use clinical, professional language.
Never include em dashes. Use commas or colons instead.
Be concise. Bullet points for next steps."""


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

    return f"""The deterministic clinical rule engine has completed its evaluation. Your task is to write the clinical summary and next steps fields only.

DISPOSITION ALREADY DETERMINED: {disposition.replace('_', ' ').upper()}
RISK LEVEL: {risk_level.upper()}
RISK SCORE: {risk_score}/15
RCRI SCORE: {rcri_score}/6

PATIENT:
Age: {age} | Sex: {sex}
Active conditions: {conditions_text}
Active medications: {meds_text}
Recent vitals: {vitals_text}

TIER 1 TRIGGERS (hard escalation flags):
{triggers_text}

TIER 2 CONTRIBUTING FACTORS:
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
Write ONLY the following two fields. Do not restate the disposition or scores.

CLINICAL_SUMMARY: (2-4 sentences. State the most clinically relevant context that explains why this disposition was reached. Be specific about the key findings.)

NEXT_STEPS:
1. (most urgent action)
2. (second action)
3. (third action if applicable)
4. (fourth action if applicable)
5. (fifth action if applicable, else omit)

Use only these two sections. Professional, concise clinical language."""
