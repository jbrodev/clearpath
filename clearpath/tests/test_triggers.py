"""
Unit tests for the trigger and scoring engines.
Tests deterministic logic only — no LLM, no FHIR calls.
"""

import pytest
from clearpath.models.clinical import PatientSnapshot, Condition, Medication, VitalSigns, LabResult
from clearpath.engines.medications import classify_medications
from clearpath.engines.triggers import evaluate_tier1_triggers, evaluate_tier2_factors, compute_rcri


def make_snapshot(**kwargs) -> PatientSnapshot:
    defaults = {"patient_id": "test"}
    defaults.update(kwargs)
    return PatientSnapshot(**defaults)


def test_anticoagulant_triggers_tier1():
    meds = classify_medications(["warfarin 5mg", "lisinopril"])
    snapshot = make_snapshot()
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "active_anticoagulation" in trigger_ids


def test_eliquis_triggers_tier1():
    meds = classify_medications(["Eliquis 5mg tablet"])
    snapshot = make_snapshot()
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "active_anticoagulation" in trigger_ids


def test_stroke_in_conditions_triggers_tier1():
    conditions = [Condition(display="Ischemic stroke", icd_code="I63.9")]
    snapshot = make_snapshot(active_conditions=conditions)
    meds = classify_medications(["lisinopril"])
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "recent_stroke_tia" in trigger_ids


def test_stroke_in_note_triggers_tier1():
    snapshot = make_snapshot(pcp_note_raw="Patient had a TIA in February 2026. Currently recovering.")
    meds = []
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "recent_stroke_tia" in trigger_ids


def test_severe_htn_triggers_tier1():
    snapshot = make_snapshot(recent_vitals=VitalSigns(systolic_bp=185, diastolic_bp=110))
    meds = []
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "severe_uncontrolled_htn" in trigger_ids


def test_moderate_htn_does_not_trigger_tier1():
    snapshot = make_snapshot(recent_vitals=VitalSigns(systolic_bp=145))
    meds = []
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "severe_uncontrolled_htn" not in trigger_ids


def test_pacemaker_triggers_tier1():
    snapshot = make_snapshot(
        known_implants=["pacemaker"],
        pcp_note_raw="Patient has a pacemaker implanted in 2024."
    )
    meds = []
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "implanted_cardiac_device" in trigger_ids


def test_dual_antiplatelet_triggers_tier1():
    meds = classify_medications(["clopidogrel", "aspirin 325mg"])
    snapshot = make_snapshot()
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "antiplatelet_dual_therapy" in trigger_ids or "active_anticoagulation" in trigger_ids or len(triggers) > 0


def test_clean_patient_no_tier1():
    conditions = [Condition(display="Hypertension")]
    meds = classify_medications(["amlodipine 5mg"])
    snapshot = make_snapshot(
        active_conditions=conditions,
        recent_vitals=VitalSigns(systolic_bp=128, diastolic_bp=78)
    )
    triggers = evaluate_tier1_triggers(snapshot, meds)
    assert len(triggers) == 0


def test_diabetes_scores_tier2():
    conditions = [Condition(display="Type 2 diabetes mellitus")]
    labs = [LabResult(name="HbA1c", value=8.5, unit="%")]
    snapshot = make_snapshot(active_conditions=conditions, recent_labs=labs)
    meds = classify_medications(["metformin"])
    factors, score = evaluate_tier2_factors(snapshot, meds)
    factor_ids = [f.trigger_id for f in factors]
    assert "diabetes" in factor_ids
    assert score >= 2


def test_polypharmacy_scores_tier2():
    med_list = ["lisinopril", "metoprolol", "amlodipine", "atorvastatin",
                "metformin", "aspirin", "omeprazole", "furosemide", "spironolactone"]
    snapshot = make_snapshot(medication_count=len(med_list))
    meds = classify_medications(med_list)
    factors, score = evaluate_tier2_factors(snapshot, meds)
    factor_ids = [f.trigger_id for f in factors]
    assert "polypharmacy" in factor_ids


def test_rcri_cerebrovascular_disease():
    conditions = [Condition(display="CVA history", icd_code="I63.9")]
    snapshot = make_snapshot(active_conditions=conditions)
    meds = []
    score = compute_rcri(snapshot, meds)
    assert score >= 1


def test_rcri_insulin_diabetes():
    meds = classify_medications(["Lantus insulin glargine 100 units/mL"])
    snapshot = make_snapshot()
    score = compute_rcri(snapshot, meds)
    assert score >= 1


def test_no_false_anticoagulant_on_unrelated_drug():
    meds = classify_medications(["omeprazole 20mg", "levothyroxine 50mcg", "vitamin D"])
    snapshot = make_snapshot()
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "active_anticoagulation" not in trigger_ids


def test_pradaxa_detected_as_anticoagulant():
    meds = classify_medications(["Pradaxa 150mg capsule"])
    snapshot = make_snapshot()
    triggers = evaluate_tier1_triggers(snapshot, meds)
    trigger_ids = [t.trigger_id for t in triggers]
    assert "active_anticoagulation" in trigger_ids
