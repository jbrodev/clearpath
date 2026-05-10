"""
Serves the A2A Agent Card at /.well-known/agent-card.json.
This is the first thing Prompt Opinion reads when you register the external agent.
Conforms to A2A v1 (https://docs.promptopinion.ai/a2a-v1-migration):
  - extensions live under capabilities.extensions[]
  - no top-level url, preferredTransport, or capabilities.stateTransitionHistory
  - FHIR context extension declares its required scopes via params.scopes[]
"""

from clearpath.models.a2a import FHIR_CONTEXT_EXTENSION_URI


_FHIR_SCOPES = [
    {"name": "patient/Patient.rs", "required": True},
    {"name": "patient/Condition.rs", "required": True},
    {"name": "patient/MedicationRequest.rs", "required": True},
    {"name": "patient/Procedure.rs", "required": True},
    {"name": "patient/DocumentReference.rs", "required": True},
    {"name": "patient/Observation.rs", "required": True},
    {"name": "patient/Encounter.rs", "required": False},
    {"name": "patient/AllergyIntolerance.rs", "required": False},
]


AGENT_CARD = {
    "name": "ClearPath",
    "description": (
        "Pre-operative anesthesia clearance triage agent. Analyzes patient chart data including "
        "active medications (generic and brand names), clinical notes, specialist history, and "
        "FHIR resources to determine whether specialist clearance is needed before anesthesia. "
        "Returns a structured disposition with risk score, triggering factors, recommended specialties, "
        "and actionable next steps. All output is for clinician review only."
    ),
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
    "skills": [
        {
            "id": "preop-clearance-triage",
            "name": "Pre-Op Clearance Triage",
            "description": (
                "Reviews patient chart context and returns a structured clearance disposition: "
                "no_clearance_needed, clearance_recommended, specialist_required, "
                "anesthesia_review_required, or insufficient_information. "
                "Includes risk score, RCRI score, triggering factors, and recommended next steps."
            ),
            "tags": [
                "preoperative",
                "clearance",
                "anesthesia",
                "perioperative",
                "risk-stratification",
                "fhir",
                "clinical-decision-support",
            ],
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Does this patient need clearance before surgery?",
                "Review this patient chart for pre-anesthesia clearance requirements",
                "What specialist clearance is needed before this procedure?",
            ],
        }
    ],
    "authentication": {"schemes": ["Bearer"]},
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "extensions": [
            {
                "uri": FHIR_CONTEXT_EXTENSION_URI,
                "description": (
                    "FHIR context allowing the agent to query a patient's chart on the workspace "
                    "FHIR server. Required: ClearPath cannot perform a clearance assessment without it."
                ),
                "required": True,
                "params": {"scopes": _FHIR_SCOPES},
            }
        ],
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
}


def get_agent_card() -> dict:
    return dict(AGENT_CARD)
