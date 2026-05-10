"""
ClearPath main processing pipeline.
Orchestrates all layers: FHIR fetch, normalization, trigger evaluation, decision, reasoning.
"""

from clearpath.fhir.client import FHIRClient, TokenExpiredError
from clearpath.fhir.normalizer import build_snapshot
from clearpath.engines.medications import classify_medications
from clearpath.engines.triggers import evaluate_tier1_triggers, evaluate_tier2_factors, compute_rcri
from clearpath.engines.decision import build_clearance_output
from clearpath.reasoning.engine import enrich_with_reasoning
from clearpath.models.a2a import FHIRContext
from clearpath.models.clinical import (
    ClearanceOutput, Disposition, RiskLevel, PatientSnapshot,
    ScoreResult, VitalSigns
)


async def run_clearance_pipeline(
    fhir_context: FHIRContext | None,
    user_query: str,
) -> ClearanceOutput:
    """
    Full pipeline from FHIR context to structured clearance output.
    Handles missing FHIR context, expired tokens, and sparse data gracefully.
    """

    # No FHIR context — return insufficient information immediately
    if not fhir_context or not fhir_context.patientId:
        output = _no_fhir_output()
        output = await enrich_with_reasoning(output, _empty_snapshot(), [], user_query)
        return output

    # Expired token check
    try:
        client = FHIRClient(fhir_context)
    except TokenExpiredError:
        output = _token_expired_output()
        output = await enrich_with_reasoning(output, _empty_snapshot(), [], user_query)
        return output
    except Exception as e:
        print(f"[clearpath] FHIR init error: {type(e).__name__}: {e}")
        output = _no_fhir_output()
        output.missing_information.append(f"FHIR initialization error: {str(e)}")
        output = await enrich_with_reasoning(output, _empty_snapshot(), [], user_query)
        return output

    # Fetch FHIR data
    fhir_data = await client.fetch_all()

    # Build normalized snapshot
    snapshot = build_snapshot(fhir_data)

    # Classify medications
    raw_med_names = [m.name for m in snapshot.active_medications]
    classified_meds = classify_medications(raw_med_names)
    snapshot.active_medications = classified_meds

    # Evaluate triggers
    tier1_triggers = evaluate_tier1_triggers(snapshot, classified_meds)
    tier2_factors, tier2_score = evaluate_tier2_factors(snapshot, classified_meds)
    rcri_score = compute_rcri(snapshot, classified_meds)

    score_result = ScoreResult(
        total_score=tier2_score,
        rcri_score=rcri_score,
        tier1_triggers=tier1_triggers,
        tier2_factors=tier2_factors,
    )

    # Build deterministic output
    output = build_clearance_output(snapshot, score_result)

    # Enrich with LLM reasoning
    output = await enrich_with_reasoning(output, snapshot, tier2_factors, user_query)

    return output


def _empty_snapshot() -> PatientSnapshot:
    return PatientSnapshot(patient_id="unknown")


def _no_fhir_output() -> ClearanceOutput:
    return ClearanceOutput(
        disposition=Disposition.INSUFFICIENT_INFORMATION,
        risk_level=RiskLevel.LOW,
        risk_score=0,
        rcri_score=0,
        confidence=0.40,
        recommended_specialties=[],
        triggering_factors=[],
        clinical_summary="",
        recommended_next_steps=[],
        missing_information=[
            "Patient FHIR context was not provided or patient ID is missing",
            "Active medication list",
            "Active condition list",
            "Primary care clinical notes",
        ],
    )


def _token_expired_output() -> ClearanceOutput:
    return ClearanceOutput(
        disposition=Disposition.INSUFFICIENT_INFORMATION,
        risk_level=RiskLevel.LOW,
        risk_score=0,
        rcri_score=0,
        confidence=0.40,
        recommended_specialties=[],
        triggering_factors=[],
        clinical_summary="",
        recommended_next_steps=[],
        missing_information=[
            "FHIR access token has expired. Please refresh your session and retry.",
        ],
    )
