"""
Trigger engine for ClearPath.
Pure Python, no LLM.
Evaluates Tier 1 (hard escalation) and Tier 2 (scoring) triggers.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

# Short abbreviations that must be matched as whole words only
_WHOLE_WORD_REQUIRED = {"tia", "mi", "cva", "chf", "cad", "osa", "acs", "icd", "vad", "dbs", "crt", "ckd", "dm", "htn", "gp", "fm", "im", "fp", "stemi", "nstemi"}

# Negation phrases — if any appear within 60 chars before a keyword match, skip it
_NEGATION_PREFIXES = (
    "no history of", "no prior", "denies", "negative for", "without",
    "ruled out", "no ", "never had", "no evidence of", "no known",
)

from clearpath.engines.medications import (
    has_anticoagulant, has_antiplatelet, has_insulin,
    is_dual_antiplatelet_or_combination, classify_medications
)
from clearpath.models.clinical import PatientSnapshot, TriggerResult, Medication


_RULES: dict | None = None


def _load_rules() -> dict:
    global _RULES
    if _RULES is None:
        path = Path(__file__).parent.parent / "data" / "trigger_rules.json"
        with open(path) as f:
            _RULES = json.load(f)
    return _RULES


def _is_negated(text_lower: str, match_start: int) -> bool:
    """Return True if a keyword match at match_start is preceded by a negation phrase."""
    window = text_lower[max(0, match_start - 60):match_start]
    return any(neg in window for neg in _NEGATION_PREFIXES)


def _keywords_in_text(keywords: list[str], text: str) -> str | None:
    """Return the first matching keyword found in text, or None.
    Short abbreviations use word boundaries to prevent false positives
    (e.g. 'tia' must not match inside 'anticoagulation').
    Skips matches that are preceded by negation phrases.
    """
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in _WHOLE_WORD_REQUIRED:
            for m in re.finditer(r"\b" + re.escape(kw_lower) + r"\b", text_lower):
                if not _is_negated(text_lower, m.start()):
                    return kw
        else:
            idx = 0
            while True:
                pos = text_lower.find(kw_lower, idx)
                if pos == -1:
                    break
                if not _is_negated(text_lower, pos):
                    return kw
                idx = pos + 1
    return None


def _icd_match(icd_prefixes: list[str], conditions) -> str | None:
    for cond in conditions:
        if cond.icd_code:
            for prefix in icd_prefixes:
                if cond.icd_code.startswith(prefix):
                    return cond.display
        display_lower = cond.display.lower()
    return None


def _get_note_text(snapshot: PatientSnapshot) -> str:
    parts = []
    if snapshot.pcp_note_raw:
        parts.append(snapshot.pcp_note_raw)
    for sp_data in snapshot.specialist_notes:
        for note in sp_data.get("notes", []):
            parts.append(note.get("text", ""))
    return " ".join(parts).lower()


def evaluate_tier1_triggers(snapshot: PatientSnapshot, classified_meds: list[Medication]) -> list[TriggerResult]:
    triggers = []
    rules = _load_rules()
    note_text = _get_note_text(snapshot)
    condition_displays = " ".join(c.display.lower() for c in snapshot.active_conditions)

    # Immunosuppressant / biologic therapy — requires rheumatology perioperative guidance
    immunosuppressant_meds = [m.name for m in classified_meds if m.flag == "immunosuppressant_therapy"]
    if immunosuppressant_meds:
        triggers.append(TriggerResult(
            trigger_id="immunosuppressant_therapy",
            label="Active immunosuppressant or biologic therapy",
            tier=1,
            specialties=["rheumatology"],
            evidence=f"Medication: {', '.join(immunosuppressant_meds[:2])}"
        ))

    # Anticoagulant — specialty depends on the indication. AFib/valvular causes
    # are managed by cardiology (they own the perioperative hold decision).
    # Thromboembolic causes (DVT/PE/thrombophilia) are managed by hematology.
    if has_anticoagulant(classified_meds):
        ac_names = [m.name for m in classified_meds if m.flag == "active_anticoagulation"]
        condition_text_lower = condition_displays.lower()
        cardiac_indication = any(kw in condition_text_lower for kw in (
            "atrial fibrillation", "afib", "a-fib", "atrial flutter",
            "valvular", "prosthetic valve", "mechanical valve",
        ))
        triggers.append(TriggerResult(
            trigger_id="active_anticoagulation",
            label="Active anticoagulation therapy",
            tier=1,
            specialties=["cardiology"] if cardiac_indication else ["hematology"],
            evidence=f"Medication: {', '.join(ac_names)}"
        ))

    # Dual antiplatelet or anticoag + antiplatelet
    if is_dual_antiplatelet_or_combination(classified_meds):
        triggers.append(TriggerResult(
            trigger_id="antiplatelet_dual_therapy",
            label="Dual antiplatelet or combined anticoag/antiplatelet therapy",
            tier=1,
            specialties=["hematology", "cardiology"],
            evidence="Two or more antiplatelet/anticoagulant agents detected"
        ))

    # Recent stroke / TIA — must filter out the AFib management phrase "stroke
    # prevention" (and similar). A patient on anticoagulation for AFib commonly
    # has "stroke prevention" / "prophylaxis" in their notes, which is NOT a
    # history of stroke. Only fire if (a) an ICD code matches, or (b) the
    # keyword appears outside a prevention/prophylaxis context.
    stroke_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "recent_stroke_tia")
    stroke_icd = _icd_match(stroke_rule.get("icd_codes", []), snapshot.active_conditions)
    combined_text = note_text + " " + condition_displays
    stroke_kw = _keywords_in_text(stroke_rule["keywords"], combined_text)

    if stroke_kw and not stroke_icd:
        # Remove all prevention/prophylaxis phrases from the text and re-check
        # whether a stroke keyword still appears. If not, treat as false positive.
        sanitized = combined_text.lower()
        for phrase in (
            "stroke prevention", "stroke prophylaxis",
            "prevent stroke", "preventing stroke", "for stroke risk",
            "stroke risk reduction", "anticoagulation for stroke",
            "afib stroke", "atrial fibrillation stroke",
        ):
            sanitized = sanitized.replace(phrase, "")
        stroke_kw_after = _keywords_in_text(stroke_rule["keywords"], sanitized)
        if not stroke_kw_after:
            stroke_kw = None  # only mentions were prevention-context — drop

    if stroke_kw or stroke_icd:
        evidence = f"Found: '{stroke_kw or stroke_icd}'"
        triggers.append(TriggerResult(
            trigger_id="recent_stroke_tia",
            label="Recent or documented stroke/TIA",
            tier=1,
            specialties=["neurology"],
            evidence=evidence
        ))

    # Recent MI
    mi_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "recent_mi")
    mi_kw = _keywords_in_text(mi_rule["keywords"], note_text + " " + condition_displays)
    mi_icd = _icd_match(mi_rule.get("icd_codes", []), snapshot.active_conditions)
    if mi_kw or mi_icd:
        triggers.append(TriggerResult(
            trigger_id="recent_mi",
            label="Recent myocardial infarction",
            tier=1,
            specialties=["cardiology"],
            evidence=f"Found: '{mi_kw or mi_icd}'"
        ))

    # Unstable angina
    angina_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "unstable_angina")
    angina_kw = _keywords_in_text(angina_rule["keywords"], note_text)
    if angina_kw:
        triggers.append(TriggerResult(
            trigger_id="unstable_angina",
            label="Unstable angina or active chest pain",
            tier=1,
            specialties=["cardiology"],
            evidence=f"Found: '{angina_kw}'"
        ))

    # Decompensated CHF
    chf_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "decompensated_chf")
    chf_kw = _keywords_in_text(chf_rule["keywords"], note_text + " " + condition_displays)
    chf_icd = _icd_match(chf_rule.get("icd_codes", []), snapshot.active_conditions)
    if chf_kw or chf_icd:
        triggers.append(TriggerResult(
            trigger_id="decompensated_chf",
            label="Decompensated or severe heart failure",
            tier=1,
            specialties=["cardiology"],
            evidence=f"Found: '{chf_kw or chf_icd}'"
        ))

    # Oxygen dependent
    o2_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "oxygen_dependent")
    o2_kw = _keywords_in_text(o2_rule["keywords"], note_text + " " + condition_displays)
    low_sat = snapshot.recent_vitals and snapshot.recent_vitals.o2_saturation and snapshot.recent_vitals.o2_saturation < 92
    if o2_kw or low_sat:
        evidence = f"Found: '{o2_kw}'" if o2_kw else f"O2 saturation: {snapshot.recent_vitals.o2_saturation}%"
        triggers.append(TriggerResult(
            trigger_id="oxygen_dependent",
            label="Oxygen-dependent or respiratory failure",
            tier=1,
            specialties=["pulmonology"],
            evidence=evidence
        ))

    # Prior anesthesia complication
    anesthesia_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "prior_anesthesia_complication")
    anesthesia_kw = _keywords_in_text(anesthesia_rule["keywords"], note_text)
    if anesthesia_kw:
        triggers.append(TriggerResult(
            trigger_id="prior_anesthesia_complication",
            label="Documented prior anesthesia complication",
            tier=1,
            specialties=["anesthesia"],
            evidence=f"Found: '{anesthesia_kw}'"
        ))

    # Severe uncontrolled hypertension
    if snapshot.recent_vitals and snapshot.recent_vitals.systolic_bp and snapshot.recent_vitals.systolic_bp >= 180:
        triggers.append(TriggerResult(
            trigger_id="severe_uncontrolled_htn",
            label=f"Severe uncontrolled hypertension (SBP {snapshot.recent_vitals.systolic_bp} mmHg)",
            tier=1,
            specialties=["cardiology"],
            evidence=f"SBP: {snapshot.recent_vitals.systolic_bp} mmHg"
        ))

    # Uncontrolled seizure disorder
    seizure_rule = next(r for r in rules["tier1_triggers"] if r["id"] == "uncontrolled_seizure_disorder")
    seizure_kw = _keywords_in_text(seizure_rule["keywords"], note_text + " " + condition_displays)
    seizure_icd = _icd_match(seizure_rule.get("icd_codes", []), snapshot.active_conditions)
    if seizure_kw or seizure_icd:
        triggers.append(TriggerResult(
            trigger_id="uncontrolled_seizure_disorder",
            label="Seizure disorder",
            tier=1,
            specialties=["neurology"],
            evidence=f"Found: '{seizure_kw or seizure_icd}'"
        ))

    # Implanted cardiac device
    implant_keywords = ["pacemaker", "icd", "implantable cardioverter", "crt-d", "crt-p", "cardiac resynchronization"]
    implant_kw = _keywords_in_text(implant_keywords, note_text + " " + " ".join(snapshot.known_implants))
    if implant_kw or any("pacemaker" in i or "icd" in i for i in snapshot.known_implants):
        triggers.append(TriggerResult(
            trigger_id="implanted_cardiac_device",
            label="Implanted cardiac device requiring evaluation",
            tier=1,
            specialties=["cardiology"],
            evidence=f"Found: '{implant_kw or snapshot.known_implants[0]}'"
        ))

    # Deduplicate by trigger_id
    seen = set()
    deduped = []
    for t in triggers:
        if t.trigger_id not in seen:
            seen.add(t.trigger_id)
            deduped.append(t)
    return deduped


def evaluate_tier2_factors(snapshot: PatientSnapshot, classified_meds: list[Medication]) -> tuple[list[TriggerResult], int]:
    factors = []
    total_score = 0
    rules = _load_rules()
    note_text = _get_note_text(snapshot)
    condition_displays = " ".join(c.display.lower() for c in snapshot.active_conditions)

    # Diabetes
    dm_keywords = ["diabetes", "dm", "diabetic"]
    hba1c = next((l.value for l in snapshot.recent_labs if "hba1c" in l.name.lower() or "hemoglobin a1c" in l.name.lower()), None)
    has_insulin_flag = has_insulin(classified_meds)

    dm_in_text = _keywords_in_text(dm_keywords, condition_displays)
    dm_icd = _icd_match(["E10", "E11", "E13"], snapshot.active_conditions)

    if dm_in_text or dm_icd:
        uncontrolled = (hba1c and hba1c >= 8.0) or has_insulin_flag
        if uncontrolled:
            pts = 2
            label = f"Uncontrolled diabetes (HbA1c: {hba1c:.1f}%)" if hba1c else "Insulin-dependent diabetes"
        else:
            pts = 1
            label = "Diabetes mellitus (controlled)"
        factors.append(TriggerResult(trigger_id="diabetes", label=label, tier=2, evidence=f"HbA1c: {hba1c}" if hba1c else None))
        total_score += pts

    # Hypertension
    htn_in_text = _keywords_in_text(["hypertension", "htn", "high blood pressure"], condition_displays)
    htn_icd = _icd_match(["I10"], snapshot.active_conditions)
    if (htn_in_text or htn_icd) and snapshot.recent_vitals and snapshot.recent_vitals.systolic_bp and snapshot.recent_vitals.systolic_bp < 180:
        factors.append(TriggerResult(trigger_id="hypertension", label="Hypertension", tier=2, evidence=f"SBP: {snapshot.recent_vitals.systolic_bp}" if snapshot.recent_vitals.systolic_bp else None))
        total_score += 1

    # COPD
    copd_in_text = _keywords_in_text(["copd", "chronic obstructive pulmonary disease", "emphysema"], condition_displays + " " + note_text)
    copd_icd = _icd_match(["J44"], snapshot.active_conditions)
    if copd_in_text or copd_icd:
        factors.append(TriggerResult(trigger_id="copd", label="COPD", tier=2, evidence=f"Found: '{copd_in_text or copd_icd}'"))
        total_score += 2

    # OSA
    osa_in_text = _keywords_in_text(["sleep apnea", "osa", "cpap", "bipap"], condition_displays + " " + note_text)
    osa_icd = _icd_match(["G47.33"], snapshot.active_conditions)
    if osa_in_text or osa_icd:
        factors.append(TriggerResult(trigger_id="osa", label="Obstructive sleep apnea", tier=2))
        total_score += 1

    # Age
    if snapshot.age and snapshot.age >= 75:
        factors.append(TriggerResult(trigger_id="advanced_age", label=f"Age {snapshot.age} (75+)", tier=2))
        total_score += 1

    # Polypharmacy
    if snapshot.medication_count >= 8:
        factors.append(TriggerResult(trigger_id="polypharmacy", label=f"Polypharmacy ({snapshot.medication_count} medications)", tier=2))
        total_score += 1

    # No recent PCP visit
    if snapshot.days_since_pcp_visit and snapshot.days_since_pcp_visit > 365:
        factors.append(TriggerResult(trigger_id="no_recent_pcp", label=f"No PCP visit in {snapshot.days_since_pcp_visit // 30} months", tier=2))
        total_score += 1

    # No recent labs
    if snapshot.days_since_any_lab and snapshot.days_since_any_lab > 180:
        factors.append(TriggerResult(trigger_id="no_recent_labs", label=f"No recent labs ({snapshot.days_since_any_lab} days ago)", tier=2))
        total_score += 1

    # Multiple comorbidities
    if snapshot.chronic_condition_count >= 4:
        factors.append(TriggerResult(trigger_id="multiple_comorbidities", label=f"Multiple comorbidities ({snapshot.chronic_condition_count} chronic conditions)", tier=2))
        total_score += 1

    # Obesity
    bmi = snapshot.recent_vitals.bmi if snapshot.recent_vitals else None
    obesity_in_conditions = _keywords_in_text(["obesity", "obese", "morbid obesity"], condition_displays)
    if (bmi and bmi >= 40) or obesity_in_conditions:
        label = f"Morbid obesity (BMI {bmi:.1f})" if bmi else "Morbid obesity"
        factors.append(TriggerResult(trigger_id="obesity", label=label, tier=2, evidence=f"BMI: {bmi}" if bmi else None))
        total_score += 1

    # CKD
    creatinine = next((l.value for l in snapshot.recent_labs if "creatinine" in l.name.lower()), None)
    ckd_in_conditions = _keywords_in_text(["chronic kidney disease", "ckd", "renal insufficiency"], condition_displays)
    ckd_icd = _icd_match(["N18.3", "N18.4", "N18.5", "N18.6"], snapshot.active_conditions)
    if ckd_in_conditions or ckd_icd or (creatinine and creatinine >= 1.5):
        label = f"CKD (creatinine {creatinine:.1f})" if creatinine else "Chronic kidney disease"
        factors.append(TriggerResult(trigger_id="ckd", label=label, tier=2))
        total_score += 1

    # Recent hospitalization
    if snapshot.days_since_pcp_visit is not None and snapshot.days_since_pcp_visit < 60:
        hosp_in_notes = _keywords_in_text(["hospitalized", "admitted", "admission", "inpatient", "hospital stay"], note_text)
        if hosp_in_notes:
            factors.append(TriggerResult(trigger_id="recent_hospitalization", label="Recent hospitalization", tier=2))
            total_score += 1

    return factors, total_score


def compute_rcri(snapshot: PatientSnapshot, classified_meds: list[Medication]) -> int:
    """Compute Revised Cardiac Risk Index score."""
    score = 0
    note_text = _get_note_text(snapshot)
    condition_displays = " ".join(c.display.lower() for c in snapshot.active_conditions)

    # High risk surgery — not determinable from chart alone, assume if procedure context given
    # We skip this criterion unless explicitly mentioned

    # Ischemic heart disease
    ihd_kw = _keywords_in_text(["coronary artery disease", "cad", "angina", "myocardial infarction", "coronary disease", "stent"], condition_displays + " " + note_text)
    ihd_icd = _icd_match(["I25", "I20", "I21", "I22"], snapshot.active_conditions)
    if ihd_kw or ihd_icd:
        score += 1

    # CHF history
    chf_kw = _keywords_in_text(["heart failure", "chf", "congestive heart failure"], condition_displays + " " + note_text)
    chf_icd = _icd_match(["I50"], snapshot.active_conditions)
    if chf_kw or chf_icd:
        score += 1

    # Cerebrovascular disease
    cvd_kw = _keywords_in_text(["stroke", "tia", "cva", "cerebrovascular"], condition_displays + " " + note_text)
    cvd_icd = _icd_match(["I63", "I64", "G45"], snapshot.active_conditions)
    if cvd_kw or cvd_icd:
        score += 1

    # Insulin-dependent diabetes
    if has_insulin(classified_meds):
        score += 1

    # Preoperative creatinine > 2.0
    creatinine = next((l.value for l in snapshot.recent_labs if "creatinine" in l.name.lower()), None)
    if creatinine and creatinine > 2.0:
        score += 1

    return score
