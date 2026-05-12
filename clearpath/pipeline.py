"""
ClearPath main processing pipeline.
Orchestrates all layers: FHIR fetch, normalization, trigger evaluation, decision, reasoning.
"""

from clearpath.fhir.client import FHIRClient, TokenExpiredError
from clearpath.fhir.normalizer import build_snapshot
from clearpath.engines.medications import classify_medications
from clearpath.engines.triggers import evaluate_tier1_triggers, evaluate_tier2_factors, compute_rcri
from clearpath.engines.decision import build_clearance_output, detect_major_procedure
from clearpath.reasoning.engine import enrich_with_reasoning, generate_clearance_letter


_LETTER_REQUEST_EXPLICIT = (
    # explicit letter phrasing
    "clearance letter", "referral letter", "approval letter",
    "clearance request letter", "letter requesting",
    "write a letter", "draft a letter", "write me a letter",
    "generate a letter", "compose a letter", "create a letter",
    "draft me a letter",
    # explicit note phrasing
    "clearance note", "referral note", "clinical note",
    "preop note", "pre-op note", "preoperative note", "pre-operative note",
    "pcp note", "medical note",
    "write a note", "draft a note", "write me a note",
    "generate a note", "compose a note", "create a note",
    "draft me a note", "note requesting", "summary note",
    # explicit request phrasing
    "clearance request", "request for clearance",
)

_VERBS = ("write", "draft", "compose", "generate", "create", "prepare", "produce")
_DOCS = ("letter", "note", "memo", "correspondence", "summary")
_CONTEXTS = (
    "pcp", "primary care", "clearance", "preop", "pre-op",
    "pre-operative", "preoperative", "surgical", "surgery",
    "anesthesia", "perioperative", "consultation",
)


def _is_letter_request(query: str) -> bool:
    """Detect any of: explicit letter/note phrases; a verb+doc near a clinical
    context; or "[doc] for/to [provider]" phrasing common in clinician requests."""
    if not query:
        return False
    q = query.lower()

    if any(kw in q for kw in _LETTER_REQUEST_EXPLICIT):
        return True

    has_verb = any(v in q for v in _VERBS)
    has_doc = any(d in q for d in _DOCS)
    has_context = any(c in q for c in _CONTEXTS)
    if has_verb and has_doc and has_context:
        return True

    # "...note/letter/summary for/to the pcp/primary care/specialist..."
    if has_doc and (" for the pcp" in q or " for the primary care" in q
                    or " to the pcp" in q or " to the primary care" in q
                    or " for her pcp" in q or " for his pcp" in q):
        return True

    return False
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

    # Detect the procedure being asked about NOW (latest mention wins if the
    # query bundles conversation history).
    current_procedure = detect_major_procedure(user_query)

    # Build deterministic output (passes user_query so the engine can detect
    # institutional-mandate procedures like cardiac/transplant surgery).
    output = build_clearance_output(snapshot, score_result, user_query=user_query)

    # Enrich with LLM reasoning. Pass the detected procedure separately so
    # Claude anchors on the current request, not stale history in the query.
    output = await enrich_with_reasoning(
        output, snapshot, tier2_factors, user_query, current_procedure
    )

    # If the user explicitly asked for a clearance letter / referral note,
    # generate one and attach it. The standard structured assessment remains.
    if _is_letter_request(user_query):
        letter = await generate_clearance_letter(
            output, snapshot, current_procedure, user_query
        )
        if letter:
            output.clearance_letter = letter

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
