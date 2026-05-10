"""
Decision engine for ClearPath.
Takes trigger results and score and produces a disposition.
Pure logic, no LLM.
"""

from datetime import datetime, timezone

from clearpath.models.clinical import (
    PatientSnapshot, Disposition, RiskLevel,
    TriggerResult, ScoreResult, SpecialistFinding, ClearanceOutput
)


# Specialties whose triggers represent high perioperative risk
_HIGH_RISK_SPECIALTIES = {"cardiology", "neurology", "pulmonology", "anesthesia", "hematology"}


def determine_risk_level(score: int, tier1_triggers: list[TriggerResult]) -> RiskLevel:
    has_high_risk_tier1 = any(
        any(s in _HIGH_RISK_SPECIALTIES for s in t.specialties)
        for t in tier1_triggers
    )
    if score >= 9:
        return RiskLevel.CRITICAL
    if has_high_risk_tier1 or score >= 6:
        return RiskLevel.HIGH
    if tier1_triggers or score >= 3:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def determine_disposition(
    tier1_triggers: list[TriggerResult],
    tier2_score: int,
    snapshot: PatientSnapshot,
    has_insufficient_data: bool,
) -> Disposition:
    if has_insufficient_data:
        return Disposition.INSUFFICIENT_INFORMATION

    if tier1_triggers:
        for t in tier1_triggers:
            if "anesthesia" in t.specialties:
                return Disposition.ANESTHESIA_REVIEW_REQUIRED
        return Disposition.SPECIALIST_REQUIRED

    if tier2_score >= 6:
        return Disposition.SPECIALIST_REQUIRED
    if tier2_score >= 3:
        return Disposition.CLEARANCE_RECOMMENDED
    return Disposition.NO_CLEARANCE_NEEDED


def get_recommended_specialties(tier1_triggers: list[TriggerResult]) -> list[str]:
    specialties = set()
    for t in tier1_triggers:
        for s in t.specialties:
            specialties.add(s)
    return sorted(specialties)


def compute_confidence(
    tier1_triggers: list[TriggerResult],
    tier2_score: int,
    snapshot: PatientSnapshot,
    has_insufficient_data: bool,
) -> float:
    if has_insufficient_data:
        return 0.45

    warnings = len(snapshot.extraction_warnings)
    has_pcp_note = bool(snapshot.pcp_note_raw)
    has_labs = bool(snapshot.recent_labs)
    has_vitals = bool(snapshot.recent_vitals and snapshot.recent_vitals.systolic_bp)
    has_conditions = bool(snapshot.active_conditions)
    has_meds = bool(snapshot.active_medications)

    data_quality = sum([has_pcp_note, has_labs, has_vitals, has_conditions, has_meds]) / 5.0
    warning_penalty = min(warnings * 0.05, 0.25)

    if tier1_triggers:
        base_confidence = 0.85 + (len(tier1_triggers) * 0.02)
    elif tier2_score >= 3:
        base_confidence = 0.75
    else:
        base_confidence = 0.80

    confidence = (base_confidence * data_quality) - warning_penalty
    return round(min(max(confidence, 0.35), 0.97), 2)


def _check_insufficient_data(snapshot: PatientSnapshot) -> tuple[bool, list[str]]:
    missing = []

    if not snapshot.active_conditions:
        missing.append("Active condition list")
    if not snapshot.active_medications:
        missing.append("Active medication list")
    if not snapshot.pcp_note_raw:
        missing.append("Primary care clinical notes")
    if not snapshot.recent_vitals or not snapshot.recent_vitals.systolic_bp:
        missing.append("Recent blood pressure readings")

    critical_missing = (
        not snapshot.active_conditions and
        not snapshot.active_medications and
        not snapshot.pcp_note_raw
    )

    return critical_missing, missing


def _build_specialist_findings(snapshot: PatientSnapshot) -> list[SpecialistFinding]:
    findings = []
    for sp_data in snapshot.specialist_notes:
        specialty = sp_data.get("specialty", "unknown")
        notes = sp_data.get("notes", [])
        if not notes:
            continue
        most_recent = notes[0]
        date_str = most_recent.get("date")
        days_ago = None
        if date_str:
            try:
                d = datetime.fromisoformat(date_str[:10]).replace(tzinfo=timezone.utc)
                days_ago = (datetime.now(timezone.utc) - d).days
            except Exception:
                pass

        status = "recent_visit" if days_ago and days_ago <= 180 else "older_visit"
        summary_text = most_recent.get("text", "")[:200]
        doctor_name = most_recent.get("doctor_name")

        findings.append(SpecialistFinding(
            specialty=specialty,
            last_visit_days_ago=days_ago,
            status=status,
            summary=summary_text or f"Note available from {specialty}",
            doctor_name=doctor_name,
        ))
    return findings


def _build_missing_info_list(snapshot: PatientSnapshot, missing_from_data: list[str]) -> list[str]:
    missing = list(missing_from_data)

    if snapshot.days_since_any_lab and snapshot.days_since_any_lab > 180:
        missing.append(f"Recent lab work (last labs {snapshot.days_since_any_lab} days ago)")

    has_creatinine = any("creatinine" in l.name.lower() for l in snapshot.recent_labs)
    has_cbc = any("cbc" in l.name.lower() or "hemoglobin" in l.name.lower() or "platelet" in l.name.lower() for l in snapshot.recent_labs)
    has_coag = any("pt" in l.name.lower() or "inr" in l.name.lower() or "ptt" in l.name.lower() for l in snapshot.recent_labs)
    has_hba1c = any("hba1c" in l.name.lower() or "hemoglobin a1c" in l.name.lower() for l in snapshot.recent_labs)

    if not has_creatinine:
        missing.append("Basic metabolic panel / creatinine")
    if not has_cbc:
        missing.append("Complete blood count")
    if not has_coag:
        for m in snapshot.active_medications:
            if m.flag == "active_anticoagulation":
                missing.append("Coagulation studies (PT/INR/aPTT)")
                break

    return missing


def build_clearance_output(
    snapshot: PatientSnapshot,
    score_result: ScoreResult,
) -> ClearanceOutput:
    has_insufficient_data, missing_from_data = _check_insufficient_data(snapshot)

    disposition = determine_disposition(
        score_result.tier1_triggers,
        score_result.total_score,
        snapshot,
        has_insufficient_data,
    )

    risk_level = determine_risk_level(score_result.total_score, score_result.tier1_triggers)
    confidence = compute_confidence(
        score_result.tier1_triggers,
        score_result.total_score,
        snapshot,
        has_insufficient_data,
    )

    specialties = get_recommended_specialties(score_result.tier1_triggers)

    triggering_factors = [t.label for t in score_result.tier1_triggers]
    triggering_factors += [t.label for t in score_result.tier2_factors if score_result.total_score >= 3]

    specialist_findings = _build_specialist_findings(snapshot)
    missing_info = _build_missing_info_list(snapshot, missing_from_data)

    return ClearanceOutput(
        disposition=disposition,
        risk_level=risk_level,
        risk_score=score_result.total_score,
        rcri_score=score_result.rcri_score,
        confidence=confidence,
        recommended_specialties=specialties,
        triggering_factors=triggering_factors,
        active_medications=[m.name for m in snapshot.active_medications],
        clinical_summary="",  # filled by reasoning engine
        recommended_next_steps=[],  # filled by reasoning engine
        specialist_findings=specialist_findings,
        missing_information=missing_info,
    )
