"""
End-to-end pipeline tests using synthetic patient FHIR bundles.
Tests the full pipeline without LLM (mocks the reasoning engine).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from clearpath.fhir.normalizer import build_snapshot
from clearpath.engines.medications import classify_medications
from clearpath.engines.triggers import evaluate_tier1_triggers, evaluate_tier2_factors, compute_rcri
from clearpath.engines.decision import build_clearance_output
from clearpath.models.clinical import Disposition, ScoreResult, ClearanceOutput


def load_synthetic_patient(filename: str) -> dict:
    path = Path(__file__).parent.parent / "data" / "synthetic_patients" / filename
    with open(path) as f:
        return json.load(f)


async def run_deterministic_pipeline(fhir_data: dict) -> ClearanceOutput:
    snapshot = build_snapshot(fhir_data)
    raw_med_names = [m.name for m in snapshot.active_medications]
    classified_meds = classify_medications(raw_med_names)
    snapshot.active_medications = classified_meds

    tier1_triggers = evaluate_tier1_triggers(snapshot, classified_meds)
    tier2_factors, tier2_score = evaluate_tier2_factors(snapshot, classified_meds)
    rcri_score = compute_rcri(snapshot, classified_meds)

    score_result = ScoreResult(
        total_score=tier2_score,
        rcri_score=rcri_score,
        tier1_triggers=tier1_triggers,
        tier2_factors=tier2_factors,
    )
    output = build_clearance_output(snapshot, score_result)
    output.clinical_summary = "Test summary"
    output.recommended_next_steps = ["Test step"]
    return output


@pytest.mark.asyncio
async def test_patient_a_high_risk():
    """Warfarin + stroke = specialist_required"""
    fhir_data = load_synthetic_patient("patient_a_high_risk.json")
    output = await run_deterministic_pipeline(fhir_data)

    assert output.disposition == Disposition.SPECIALIST_REQUIRED
    assert output.risk_level.value in ("high", "critical")
    assert "hematology" in output.recommended_specialties or "neurology" in output.recommended_specialties
    trigger_ids_raw = " ".join(output.triggering_factors).lower()
    assert "anticoag" in trigger_ids_raw or "warfarin" in trigger_ids_raw or "stroke" in trigger_ids_raw


@pytest.mark.asyncio
async def test_patient_b_low_risk():
    """Well-controlled HTN only = no_clearance_needed"""
    fhir_data = load_synthetic_patient("patient_b_low_risk.json")
    output = await run_deterministic_pipeline(fhir_data)

    assert output.disposition == Disposition.NO_CLEARANCE_NEEDED
    assert output.risk_level.value in ("low", "moderate")
    assert output.risk_score <= 3


@pytest.mark.asyncio
async def test_truly_empty_chart_is_insufficient():
    """Chart with no conditions, meds, or PCP note resolves to insufficient_information."""
    empty_bundle = {"resourceType": "Bundle", "type": "searchset", "entry": []}
    empty_fhir = {
        "patient": {"id": "x", "gender": "unknown"},
        "conditions": empty_bundle,
        "medications": empty_bundle,
        "vitals": empty_bundle,
        "labs": empty_bundle,
        "documents": empty_bundle,
        "encounters": empty_bundle,
        "procedures": empty_bundle,
        "allergies": empty_bundle,
    }
    output = await run_deterministic_pipeline(empty_fhir)

    assert output.disposition == Disposition.INSUFFICIENT_INFORMATION
    assert len(output.missing_information) > 0


@pytest.mark.asyncio
async def test_no_fhir_context_returns_insufficient():
    """No FHIR context at all = insufficient_information"""
    from clearpath.pipeline import run_clearance_pipeline

    with patch("clearpath.reasoning.engine.enrich_with_reasoning", new_callable=AsyncMock) as mock_reason:
        mock_reason.side_effect = lambda out, snap, factors, query: _passthrough(out)
        output = await run_clearance_pipeline(None, "test query")

    assert output.disposition == Disposition.INSUFFICIENT_INFORMATION


async def _passthrough(output):
    output.clinical_summary = "Test"
    output.recommended_next_steps = ["Test"]
    return output


def test_output_schema_fields():
    """Output always has all required fields."""
    from clearpath.models.clinical import ClearanceOutput, Disposition, RiskLevel
    output = ClearanceOutput(
        disposition=Disposition.NO_CLEARANCE_NEEDED,
        risk_level=RiskLevel.LOW,
        risk_score=0,
        rcri_score=0,
        confidence=0.85,
        clinical_summary="Test",
        recommended_next_steps=["Step 1"],
    )
    assert output.schema_version == "1.0"
    assert output.model_version == "clearpath-v1"
    assert output.generated_at.endswith("Z")


def test_markdown_output_renders():
    """to_markdown() produces non-empty string with all sections."""
    from clearpath.models.clinical import ClearanceOutput, Disposition, RiskLevel
    output = ClearanceOutput(
        disposition=Disposition.SPECIALIST_REQUIRED,
        risk_level=RiskLevel.HIGH,
        risk_score=8,
        rcri_score=3,
        confidence=0.91,
        recommended_specialties=["cardiology"],
        triggering_factors=["recent stroke", "anticoagulation"],
        clinical_summary="Patient has recent stroke on anticoagulation.",
        recommended_next_steps=["Obtain neurology clearance"],
        missing_information=["coagulation labs"],
    )
    md = output.to_markdown()
    assert "Specialist Required" in md
    assert "cardiology" in md.lower()
    assert "Risk: HIGH" in md
    assert "RCRI 3/6" in md
    assert "Flags" in md
    assert "Next Steps" in md
