"""
LLM reasoning engine for ClearPath.
Single reasoning pass via Claude API.
Parses structured output into clinical_summary and recommended_next_steps.
Falls back to deterministic summary if LLM fails or times out.
"""

import asyncio
import os
import re

import anthropic

from clearpath.models.clinical import ClearanceOutput, PatientSnapshot, Disposition
from clearpath.reasoning.templates import SYSTEM_PROMPT, build_reasoning_prompt


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=api_key)


def _parse_llm_output(text: str) -> tuple[str, list[str]]:
    """Extract clinical_summary and next_steps from LLM output."""
    summary = ""
    steps = []

    summary_match = re.search(r"CLINICAL_SUMMARY:\s*(.+?)(?=NEXT_STEPS:|$)", text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()

    steps_match = re.search(r"NEXT_STEPS:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_text = steps_match.group(1).strip()
        step_lines = re.findall(r"^\d+\.\s*(.+)$", steps_text, re.MULTILINE)
        steps = [s.strip() for s in step_lines if s.strip()]

    return summary, steps


def _fallback_summary(output: ClearanceOutput, snapshot: PatientSnapshot) -> tuple[str, list[str]]:
    """Generate deterministic summary when LLM is unavailable."""
    trigger_labels = output.triggering_factors[:3]
    cond_count = len(snapshot.active_conditions)
    med_count = snapshot.medication_count

    if output.disposition == Disposition.INSUFFICIENT_INFORMATION:
        summary = (
            f"Insufficient clinical data is available to complete a full pre-operative clearance evaluation. "
            f"Key information including clinical notes, medication history, or vital signs may be missing or inaccessible."
        )
        steps = [
            "Obtain complete medication reconciliation",
            "Retrieve most recent PCP or primary care note",
            "Obtain baseline vital signs and recent labs before scheduling",
        ]
    elif output.disposition == Disposition.NO_CLEARANCE_NEEDED:
        summary = (
            f"Patient presents with {cond_count} documented condition(s) and {med_count} active medication(s). "
            f"No high-risk triggers were identified. Risk score is low, and available clinical data does not suggest need for specialist clearance prior to anesthesia."
        )
        steps = [
            "Confirm medication list is current before procedure",
            "Ensure pre-anesthesia evaluation is completed per standard protocol",
            "Review for any new symptoms or changes since last visit",
        ]
    else:
        trigger_str = "; ".join(trigger_labels[:2]) if trigger_labels else "multiple risk factors"
        summary = (
            f"Pre-operative evaluation identified {len(output.triggering_factors)} escalating factor(s) including {trigger_str}. "
            f"Risk score of {output.risk_score} and RCRI score of {output.rcri_score} support the need for additional evaluation prior to anesthesia."
        )
        steps = ["Obtain specialist clearance as indicated", "Review perioperative medication management plan"]
        if output.missing_information:
            steps.append(f"Obtain missing clinical data: {output.missing_information[0]}")

    return summary, steps


async def enrich_with_reasoning(
    output: ClearanceOutput,
    snapshot: PatientSnapshot,
    tier2_factors: list,
    user_query: str,
) -> ClearanceOutput:
    """
    Call the LLM to generate clinical_summary and recommended_next_steps.
    Falls back to deterministic summary on any failure.
    """
    try:
        client = _get_client()

        patient_context = {
            "age": snapshot.age,
            "sex": snapshot.sex,
            "conditions": [c.display for c in snapshot.active_conditions],
            "medications": [m.name for m in snapshot.active_medications],
            "vitals": {
                "systolic_bp": snapshot.recent_vitals.systolic_bp if snapshot.recent_vitals else None,
                "diastolic_bp": snapshot.recent_vitals.diastolic_bp if snapshot.recent_vitals else None,
                "heart_rate": snapshot.recent_vitals.heart_rate if snapshot.recent_vitals else None,
                "bmi": snapshot.recent_vitals.bmi if snapshot.recent_vitals else None,
            },
            "pcp_note_excerpt": (snapshot.pcp_note_raw or "")[:500],
        }

        prompt = build_reasoning_prompt(
            disposition=output.disposition.value,
            risk_level=output.risk_level.value,
            risk_score=output.risk_score,
            rcri_score=output.rcri_score,
            triggering_factors=output.triggering_factors,
            tier2_factors=[t.label for t in tier2_factors],
            patient_context=patient_context,
            specialist_findings=[
                {"specialty": sf.specialty, "summary": sf.summary, "status": sf.status}
                for sf in output.specialist_findings
            ],
            missing_info=output.missing_information,
            user_query=user_query,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        )

        raw_text = response.content[0].text if response.content else ""
        summary, steps = _parse_llm_output(raw_text)

        if not summary:
            summary, steps = _fallback_summary(output, snapshot)

    except Exception:
        summary, steps = _fallback_summary(output, snapshot)

    output.clinical_summary = summary
    output.recommended_next_steps = steps
    return output


def _build_followup_prompt(
    output: ClearanceOutput,
    snapshot: PatientSnapshot,
    question: str,
    history: list[dict],
) -> str:
    lines = []

    # Patient identity
    name_parts = [snapshot.first_name or "", snapshot.last_name or ""]
    name = " ".join(p for p in name_parts if p).strip() or "Unknown"
    lines.append(f"PATIENT: {name} | Age: {snapshot.age or 'unknown'} | Sex: {snapshot.sex or 'unknown'}")
    lines.append("")

    # Conditions
    if snapshot.active_conditions:
        lines.append("ACTIVE CONDITIONS:")
        for c in snapshot.active_conditions:
            onset = f" — onset {c.onset_date}" if c.onset_date else ""
            icd = f" ({c.icd_code})" if c.icd_code else ""
            lines.append(f"- {c.display}{icd}{onset}")
        lines.append("")

    # Medications
    if snapshot.active_medications:
        lines.append("ACTIVE MEDICATIONS:")
        for m in snapshot.active_medications:
            lines.append(f"- {m.name}")
        lines.append("")

    # Vitals
    if snapshot.recent_vitals:
        v = snapshot.recent_vitals
        vitals_parts = []
        if v.systolic_bp and v.diastolic_bp:
            vitals_parts.append(f"BP {v.systolic_bp}/{v.diastolic_bp} mmHg")
        if v.heart_rate:
            vitals_parts.append(f"HR {v.heart_rate} bpm")
        if v.o2_saturation:
            vitals_parts.append(f"O2 {v.o2_saturation}%")
        if v.bmi:
            vitals_parts.append(f"BMI {v.bmi}")
        if vitals_parts:
            lines.append(f"RECENT VITALS: {', '.join(vitals_parts)}")
            lines.append("")

    # Labs
    if snapshot.recent_labs:
        lines.append("RECENT LABS:")
        for lab in snapshot.recent_labs:
            flag = " [ABNORMAL]" if lab.abnormal else ""
            val = f"{lab.value} {lab.unit}" if lab.value is not None else "N/A"
            date = f" ({lab.date[:10]})" if lab.date else ""
            lines.append(f"- {lab.name}: {val}{date}{flag}")
        lines.append("")

    # PCP note
    if snapshot.pcp_note_raw:
        lines.append("PCP NOTE (most recent):")
        lines.append(snapshot.pcp_note_raw[:1200])
        lines.append("")

    # Specialist notes
    if snapshot.specialist_notes:
        lines.append("SPECIALIST NOTES:")
        for sp_data in snapshot.specialist_notes:
            specialty = sp_data.get("specialty", "unknown")
            notes = sp_data.get("notes", [])
            if notes:
                note = notes[0]
                doctor = note.get("doctor_name") or ""
                date = note.get("date", "")[:10] if note.get("date") else ""
                header = f"[{specialty}]"
                if doctor:
                    header += f" — {doctor}"
                if date:
                    header += f" — {date}"
                lines.append(header + ":")
                lines.append(note.get("text", "")[:800])
                lines.append("")

    # Prior clearance assessment
    lines.append("PRIOR CLEARANCE ASSESSMENT:")
    disposition_label = output.disposition.value.replace("_", " ").title()
    if output.recommended_specialties and output.disposition.value in (
        "specialist_required", "anesthesia_review_required", "clearance_recommended"
    ):
        specs = ", ".join(s.title() for s in output.recommended_specialties)
        disposition_label += f": {specs}"
    lines.append(f"  Disposition: {disposition_label}")
    lines.append(f"  Risk Level: {output.risk_level.value.upper()}  |  RCRI: {output.rcri_score}/6")
    if output.triggering_factors:
        lines.append(f"  Triggering Factors: {'; '.join(output.triggering_factors)}")
    if output.recommended_specialties:
        lines.append(f"  Recommended Specialties: {', '.join(s.title() for s in output.recommended_specialties)}")
    if output.clinical_summary:
        lines.append(f"  Clinical Summary: {output.clinical_summary}")
    if output.recommended_next_steps:
        lines.append("  Recommended Next Steps:")
        for i, step in enumerate(output.recommended_next_steps, 1):
            lines.append(f"    {i}. {step}")
    lines.append("")

    # Conversation history
    if history:
        lines.append("PRIOR CONVERSATION:")
        for turn in history:
            lines.append(f"  Clinician: {turn['q']}")
            # Truncate long assistant turns (full reports) to keep context manageable
            a = turn["a"]
            if len(a) > 600:
                a = a[:600] + "... [report truncated]"
            lines.append(f"  ClearPath: {a}")
        lines.append("")

    lines.append(f"CURRENT QUESTION: {question}")
    return "\n".join(lines)


_FOLLOWUP_SYSTEM = (
    "You are ClearPath, a clinical decision support tool. A pre-operative clearance assessment "
    "has already been completed for this patient, and you have access to their full chart. "
    "Answer the clinician's question accurately using any available patient data — conditions, "
    "medications, labs, vitals, clinical notes, or the prior clearance assessment. "
    "Be concise, clinically precise, and cite specific sources when relevant "
    "(e.g., 'per the cardiology note from Dr. Liu'). "
    "Do not repeat the full clearance report unless explicitly asked."
)


async def answer_followup(
    output: ClearanceOutput,
    snapshot: PatientSnapshot,
    question: str,
    history: list[dict],
) -> str:
    """
    Answer a follow-up clinical question using the cached assessment + full patient record.
    Returns a conversational markdown string.
    """
    try:
        client = _get_client()
        prompt = _build_followup_prompt(output, snapshot, question, history)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=_FOLLOWUP_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return response.content[0].text.strip() if response.content else "No response generated."
    except Exception:
        return "Unable to process follow-up. Type `refresh` for a new full assessment."
