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


_sync_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _sync_client
    if _sync_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        _sync_client = anthropic.Anthropic(api_key=api_key)
    return _sync_client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        _async_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _async_client


def _demote_headers(text: str) -> str:
    """Replace any markdown headers (# ## ### ####) with bold inline emphasis
    so the rendered output stays compact and doesn't blow up section labels
    into huge headings."""
    def _replace(match: re.Match) -> str:
        label = match.group(2).strip().rstrip(":")
        return f"**{label}:**"
    return re.sub(r"^(#{1,6})\s+(.+)$", _replace, text, flags=re.MULTILINE)


def _parse_llm_output(text: str) -> tuple[str, list[str]]:
    """Extract clinical_summary and next_steps from LLM output."""
    summary = ""
    steps = []

    summary_match = re.search(r"CLINICAL_SUMMARY:\s*(.+?)(?=NEXT_STEPS:|$)", text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = _demote_headers(summary_match.group(1).strip())

    steps_match = re.search(r"NEXT_STEPS:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_text = steps_match.group(1).strip()
        step_lines = re.findall(r"^\d+\.\s*(.+)$", steps_text, re.MULTILINE)
        steps = [_demote_headers(s).strip() for s in step_lines if s.strip()]

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
    current_procedure: str | None = None,
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
            current_procedure=current_procedure,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0,
            )
        )

        raw_text = response.content[0].text if response.content else ""
        summary, steps = _parse_llm_output(raw_text)

        if not summary:
            summary, steps = _fallback_summary(output, snapshot)

    except Exception as e:
        print(f"[clearpath] reasoning fallback: {type(e).__name__}: {e}")
        summary, steps = _fallback_summary(output, snapshot)

    output.clinical_summary = summary
    output.recommended_next_steps = steps
    return output


_LETTER_SYSTEM_PROMPT = (
    "You are ClearPath. Draft a concise, professional pre-operative clearance "
    "request letter FROM the surgical office TO the patient's primary care "
    "provider (PCP). The PCP is the one who follows the patient longitudinally "
    "and will coordinate clearance (including any specialist consults). Use a "
    "standard clinical referral letter format. Keep it under 250 words.\n\n"
    "Required structure:\n"
    "- [Date] placeholder line\n"
    "- 'To:' line — use the PCP value provided in the user prompt VERBATIM. Do not leave it blank.\n"
    "- 'RE: Pre-operative Clearance Request — [Patient Name], DOB [DOB]' — use the actual patient name and DOB from the prompt; do not bracket them if they are provided.\n"
    "- Greeting: 'Dear Dr. [PCP Last Name],' if a PCP name is given, else 'Dear Primary Care Provider,'. Never write 'Dear Colleague'.\n"
    "- One short paragraph stating the scheduled procedure, that the surgical office is requesting pre-operative medical clearance, and the clinical rationale (e.g. institutional protocol for major procedures, or specific Tier 1 triggers).\n"
    "- A short bulleted list of what the surgical office is asking the PCP to do: confirm clearance, coordinate any required specialist consults (name them if applicable), document any relevant findings.\n"
    "- Brief closing line requesting the completed clearance prior to surgery.\n"
    "- Signature: use the REFERRING PROVIDER value from the prompt VERBATIM. Typically this will be a surgical office or referring surgeon placeholder.\n\n"
    "Rules:\n"
    "- The letter is FROM the surgical/referring office TO the PCP. Do NOT flip the direction.\n"
    "- Fill in every field that the prompt provides as a real value. Only use bracket placeholders for fields the prompt explicitly leaves blank (e.g. date of surgery, facility, surgeon name).\n"
    "- If a specialist consult is recommended (e.g. cardiology for anticoagulation review), name it in the body as something the PCP should coordinate, NOT as the letter's addressee.\n"
    "- Do NOT suggest the PCP 'schedule a visit with' or 'coordinate with' the team performing the surgery — that team is the one requesting the clearance and does not need to be visited again. The pre-operative evaluation is a PCP-led medical clearance, performed by the PCP.\n"
    "- When asking for a pre-op visit, frame it as a PCP visit for medical clearance (e.g., 'Conduct a brief pre-operative evaluation in your office'), not as scheduling with the surgical team.\n"
    "- Do NOT invent dates, doctor names, facility names, or guideline citations.\n"
    "- Plain prose. No preamble, no commentary, no 'Here is the letter:'. Output the letter directly.\n"
    "- Do NOT use markdown headers (no `#`, `##`, `###`). Use bold (`**`) only for the field labels like RE: and To:.\n"
    "- For clinician review only. This is a draft, not signed medical correspondence."
)


# Map detected major procedure category → default consulting specialty when
# the deterministic engine hasn't already picked one from Tier 1 triggers.
_PROCEDURE_CONSULTANT = {
    "cardiac surgery": "Cardiology",
    "major vascular surgery": "Vascular Surgery / Cardiology",
    "neurosurgery": "Neurology",
    "major thoracic surgery": "Pulmonology",
    "major abdominal surgery": "Internal Medicine",
    "organ transplant": "Transplant Medicine",
    "major orthopedic surgery": "Internal Medicine",
}


async def generate_clearance_letter(
    output: ClearanceOutput,
    snapshot: PatientSnapshot,
    current_procedure: str | None,
    user_query: str,
) -> str | None:
    """Generate a draft pre-op clearance request letter. Returns None on failure."""
    try:
        client = _get_client()

        name = " ".join(p for p in [snapshot.first_name or "", snapshot.last_name or ""] if p).strip() or "[Patient Name]"
        dob = snapshot.birth_date or "[DOB]"
        conditions = ", ".join(c.display for c in snapshot.active_conditions[:6]) or "none documented"
        meds = ", ".join(m.name for m in snapshot.active_medications[:6]) or "none documented"
        triggers = "\n".join(f"- {t}" for t in output.triggering_factors[:6]) or "- (no specific triggers — institutional protocol)"

        # Who is the letter addressed TO: the PCP (or a generic primary care
        # placeholder if no PCP doctor name is available).
        pcp_recipient = snapshot.pcp_doctor_name or "Primary Care Provider"

        # Recommended specialist consult — only populated when a Tier-1 trigger
        # genuinely requires a different specialty's input (e.g., cardiology for
        # anticoag review before colonoscopy). For pure institutional-protocol
        # escalations (e.g., healthy patient going for neurosurgery), no extra
        # consult is needed: the operating team IS the operating team, the PCP
        # just provides medical clearance.
        if output.recommended_specialties:
            specialist_consult = output.recommended_specialties[0].title()
        else:
            specialist_consult = ""

        # Letter is signed by the surgical/referring office — placeholder by
        # default since we don't know who's running ClearPath.
        signing_office = "[Surgical Office / Referring Surgeon]"
        procedure = current_procedure or "[scheduled procedure]"

        specialist_line = (
            f"Specialist consultation the PCP should coordinate (mention in body, not as addressee): {specialist_consult}"
            if specialist_consult
            else "Specialist consultation needed: None — PCP clearance alone is sufficient"
        )

        user_prompt = f"""Draft the clearance request letter using these chart facts.

PATIENT NAME: {name}
DOB: {dob}
AGE/SEX: {snapshot.age or '[age]'} / {snapshot.sex or '[sex]'}
SCHEDULED PROCEDURE: {procedure}
PCP (the letter is addressed TO this person — use VERBATIM in the To: line and greeting): {pcp_recipient}
{specialist_line}
REFERRING PROVIDER / SIGNATURE (use VERBATIM at the bottom): {signing_office}

Active conditions: {conditions}
Active medications: {meds}
Risk level: {output.risk_level.value.upper()} | RCRI: {output.rcri_score}/6

Reason clearance is being requested (use these as the bulleted items the surgical office wants the PCP to address):
{triggers}

User's request that triggered this letter: {user_query}

Output the letter directly. No preamble."""

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=_LETTER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=30.0,
            )
        )
        text = response.content[0].text.strip() if response.content else ""
        return text or None

    except Exception as e:
        print(f"[clearpath] letter generation failed: {type(e).__name__}: {e}")
        return None


def _build_chart_context(output: ClearanceOutput, snapshot: PatientSnapshot) -> str:
    """
    Static chart context for a given patient + assessment. Identical across follow-ups
    in a session, so it's a prime candidate for prompt caching.
    """
    lines = []

    name_parts = [snapshot.first_name or "", snapshot.last_name or ""]
    name = " ".join(p for p in name_parts if p).strip() or "Unknown"
    lines.append(f"PATIENT: {name} | Age: {snapshot.age or 'unknown'} | Sex: {snapshot.sex or 'unknown'}")
    lines.append("")

    if snapshot.active_conditions:
        lines.append("ACTIVE CONDITIONS:")
        for c in snapshot.active_conditions:
            onset = f" (onset {c.onset_date})" if c.onset_date else ""
            lines.append(f"- {c.display}{onset}")
        lines.append("")

    if snapshot.active_medications:
        lines.append("ACTIVE MEDICATIONS (drug class for specialty mapping):")
        for m in snapshot.active_medications:
            cls = f" [{m.drug_class}]" if m.drug_class else ""
            sp = f" (typically {m.specialty})" if m.specialty else ""
            lines.append(f"- {m.name}{cls}{sp}")
        lines.append("")

    if snapshot.recent_vitals:
        v = snapshot.recent_vitals
        parts = []
        if v.systolic_bp and v.diastolic_bp:
            parts.append(f"BP {v.systolic_bp}/{v.diastolic_bp}")
        if v.heart_rate:
            parts.append(f"HR {v.heart_rate}")
        if v.o2_saturation:
            parts.append(f"O2 {v.o2_saturation}%")
        if v.bmi:
            parts.append(f"BMI {v.bmi}")
        if parts:
            lines.append(f"RECENT VITALS: {', '.join(parts)}")
            lines.append("")

    abnormal_labs = [l for l in snapshot.recent_labs if l.abnormal]
    if abnormal_labs:
        lines.append("ABNORMAL LABS:")
        for lab in abnormal_labs:
            val = f"{lab.value} {lab.unit}" if lab.value is not None else "N/A"
            date = f" ({lab.date[:10]})" if lab.date else ""
            lines.append(f"- {lab.name}: {val}{date}")
        lines.append("")

    if snapshot.pcp_note_raw:
        lines.append("PCP NOTE:")
        lines.append(snapshot.pcp_note_raw[:600])
        lines.append("")

    if snapshot.specialist_notes:
        lines.append("SPECIALIST NOTES:")
        for sp_data in snapshot.specialist_notes:
            specialty = sp_data.get("specialty", "unknown")
            notes = sp_data.get("notes", [])
            if notes:
                note = notes[0]
                doctor = note.get("doctor_name") or ""
                header = f"[{specialty}]" + (f" {doctor}" if doctor else "")
                lines.append(header + ":")
                lines.append(note.get("text", "")[:400])
                lines.append("")

    lines.append("PRIOR CLEARANCE ASSESSMENT:")
    disposition_label = output.disposition.value.replace("_", " ").title()
    if output.recommended_specialties and output.disposition.value in (
        "specialist_required", "anesthesia_review_required", "clearance_recommended"
    ):
        disposition_label += ": " + ", ".join(s.title() for s in output.recommended_specialties)
    lines.append(f"  Disposition: {disposition_label}")
    lines.append(f"  Risk: {output.risk_level.value.upper()} | RCRI {output.rcri_score}/6")
    if output.triggering_factors:
        lines.append(f"  Flags: {'; '.join(output.triggering_factors)}")
    if output.clinical_summary:
        lines.append(f"  Summary: {output.clinical_summary}")
    if output.recommended_next_steps:
        lines.append("  Next Steps: " + " | ".join(output.recommended_next_steps))

    return "\n".join(lines)


def _build_question_block(question: str, history: list[dict]) -> str:
    """Per-turn block: short recent history + current question. NOT cached."""
    lines = []
    # Keep only the last 2 turns so cache hits more often and prompt stays small.
    recent = history[-2:] if history else []
    if recent:
        lines.append("RECENT CONVERSATION:")
        for turn in recent:
            lines.append(f"  Q: {turn['q']}")
            a = turn["a"]
            if len(a) > 400:
                a = a[:400] + "..."
            lines.append(f"  A: {a}")
        lines.append("")
    lines.append(f"CURRENT QUESTION: {question}")
    return "\n".join(lines)


_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 2,
}

_FOLLOWUP_SYSTEM = (
    "You are ClearPath. A clearance assessment has already been done for this patient and "
    "you have their full chart.\n\n"
    "Answer in plain English a patient could understand. Skip jargon. If you must use a clinical "
    "term, define it in the same sentence.\n\n"
    "Ground every clinical recommendation in a recognized standard and name it: "
    "ACC/AHA perioperative cardiovascular guidelines, ASA preoperative evaluation, "
    "RCRI for cardiac risk, STOP-BANG for OSA, ARISCAT for pulmonary risk, "
    "FDA drug labeling, or the most recent specialty-society guideline. "
    "When the question goes beyond the chart (drug mechanisms, interactions, guideline thresholds, "
    "perioperative protocols, diagnosis criteria), use the web_search tool to confirm before "
    "answering.\n\n"
    "Reasoning rules about prescribers and specialists:\n"
    "- The prescriber of a medication is whoever the chart says wrote the prescription. "
    "Do NOT assume one physician manages the entire med list just because their note appears "
    "in the chart.\n"
    "- A specialist's scope is limited to the conditions they treat. Cardiology manages "
    "cardiac drugs (beta blockers, ACE inhibitors, anticoagulants for AFib, statins for ASCVD); "
    "endocrinology manages diabetes/thyroid drugs; pulmonology manages inhalers and oxygen; "
    "the PCP typically manages everything else and is the prescriber of record unless the "
    "note explicitly says otherwise.\n"
    "- When the user asks who manages a med, map drug-by-drug from drug class to specialty. "
    "Do not lump.\n\n"
    "Format rules:\n"
    "- Lead with the direct answer in the first sentence.\n"
    "- 1 to 3 short sentences total, or up to 4 tight bullets. No preamble.\n"
    "- Do NOT use markdown headers (no `#`, `##`, `###`, etc.). Use inline bold (`**text**`) for emphasis only. Keep the response compact: no large section headings like 'What It Does' or 'Key Points'.\n"
    "- Inline-link any guideline, study, or drug label, e.g. 'per the [2022 ACC/AHA guidelines](https://...)'. "
    "Do NOT add a Sources section at the end.\n"
    "- When citing chart data, name the source briefly (e.g., 'per Dr. Liu's cardiology note').\n"
    "- Avoid em dashes; use commas or colons.\n"
    "- Do not repeat the full clearance report.\n"
    "- Never invent guidelines, dosages, prescribers, or citations. If the chart does not say "
    "who prescribed something, say so — do not guess."
)


async def stream_followup(
    output: ClearanceOutput,
    snapshot: PatientSnapshot,
    question: str,
    history: list[dict],
):
    """
    Async generator yielding response text chunks for a follow-up question.

    Optimizations:
      - Streaming: time-to-first-token is ~1s instead of waiting for the full response.
      - Prompt caching: the system prompt and patient chart context are tagged with
        cache_control so follow-ups #2+ in a session hit the Anthropic prompt cache,
        cutting input processing time and cost (~10x) for the cached portion.
      - Tighter caps: max_tokens=500, max_uses=2 on web_search, trimmed chart context.

    Yields text chunks as they arrive from the model. Caller is responsible for
    accumulating the full response if needed (e.g., to store in conversation history).
    """
    try:
        client = _get_async_client()
    except ValueError:
        yield "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        return

    chart_context = _build_chart_context(output, snapshot)
    question_block = _build_question_block(question, history)

    system = [
        {"type": "text", "text": _FOLLOWUP_SYSTEM, "cache_control": {"type": "ephemeral"}},
    ]
    user_content = [
        {"type": "text", "text": chart_context, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": question_block},
    ]

    async def _stream(use_web_search: bool):
        kwargs = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
            "timeout": 45.0,
        }
        if use_web_search:
            kwargs["tools"] = [_WEB_SEARCH_TOOL]
        return client.messages.stream(**kwargs)

    try:
        async with await _stream(True) as stream:
            async for chunk in stream.text_stream:
                yield chunk
        return
    except anthropic.BadRequestError:
        pass  # fall through to retry without web_search
    except Exception as e:
        print(f"[clearpath] followup stream error: {type(e).__name__}: {e}")
        yield "Unable to process follow-up. Type `refresh` for a new full assessment."
        return

    try:
        async with await _stream(False) as stream:
            async for chunk in stream.text_stream:
                yield chunk
    except Exception as e:
        print(f"[clearpath] followup retry error: {type(e).__name__}: {e}")
        yield "Unable to process follow-up. Type `refresh` for a new full assessment."
