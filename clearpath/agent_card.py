"""
Serves the A2A Agent Card at /.well-known/agent-card.json.
This is the first thing Prompt Opinion reads when you register the external agent.
Must return valid JSON matching A2A v0.3 spec.
"""

import os

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
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Does this patient need clearance before surgery?",
                "Review this patient chart for pre-anesthesia clearance requirements",
                "What specialist clearance is needed before this procedure?"
            ]
        }
    ],
    "authentication": {
        "schemes": ["Bearer"]
    },
    "extensions": [
        {
            "uri": "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context",
            "description": "FHIR context providing access to patient data from the workspace FHIR server",
            "required": True
        }
    ],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"]
}


def get_agent_card() -> dict:
    card = dict(AGENT_CARD)
    base_url = os.environ.get("CLEARPATH_BASE_URL", "").rstrip("/")
    if base_url:
        card["url"] = base_url
    return card
