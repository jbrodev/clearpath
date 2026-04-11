"""
Extracts text from FHIR DocumentReference resources.
Prioritizes PCP/Family Medicine/Internal Medicine notes.
Also extracts specialist notes from the last 6 months.
"""

import base64
import re
from datetime import datetime, timedelta, timezone


PCP_SPECIALTIES = {
    "family medicine", "family practice", "internal medicine",
    "general practice", "primary care", "general internal medicine",
    "fm", "im", "fp", "gp", "internist", "family physician"
}

SPECIALIST_SPECIALTIES = {
    "cardiology", "cardiologist", "cardiac",
    "neurology", "neurologist", "neuroscience",
    "hematology", "hematologist",
    "pulmonology", "pulmonologist", "pulmonary",
    "nephrology", "nephrologist",
    "endocrinology", "endocrinologist",
    "gastroenterology", "gastroenterologist",
    "rheumatology", "rheumatologist",
    "oncology", "oncologist",
    "infectious disease",
    "sleep medicine",
}


def _extract_text_from_doc(doc: dict) -> str:
    """Extract plain text from a DocumentReference resource."""
    content_list = doc.get("content", [])
    for content in content_list:
        attachment = content.get("attachment", {})

        if "data" in attachment:
            try:
                decoded = base64.b64decode(attachment["data"]).decode("utf-8", errors="replace")
                return _clean_text(decoded)
            except Exception:
                pass

        if "url" in attachment:
            pass

        if "title" in attachment:
            return attachment["title"]

    # Fall back to text narrative
    if "text" in doc:
        div = doc["text"].get("div", "")
        clean = re.sub(r"<[^>]+>", " ", div)
        return _clean_text(clean)

    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_doc_date(doc: dict) -> datetime | None:
    date_str = doc.get("date") or doc.get("indexed")
    if not date_str:
        ctx = doc.get("context", {})
        period = ctx.get("period", {})
        date_str = period.get("start") or period.get("end")
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:10]).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _get_doc_specialty(doc: dict) -> str:
    """Best-effort extraction of the authoring specialty from a DocumentReference."""
    type_text = ""
    doc_type = doc.get("type", {})
    for coding in doc_type.get("coding", []):
        type_text += " " + coding.get("display", "").lower()
    type_text += " " + doc_type.get("text", "").lower()

    category_text = ""
    for cat in doc.get("category", []):
        for coding in cat.get("coding", []):
            category_text += " " + coding.get("display", "").lower()
        category_text += " " + cat.get("text", "").lower()

    context = doc.get("context", {})
    practice_setting = context.get("practiceSetting", {})
    practice_text = ""
    for coding in practice_setting.get("coding", []):
        practice_text += " " + coding.get("display", "").lower()
    practice_text += " " + practice_setting.get("text", "").lower()

    combined = type_text + " " + category_text + " " + practice_text
    return combined.strip()


def _classify_note(doc: dict) -> str:
    """Returns 'pcp', a specialist name, or 'other'."""
    specialty_text = _get_doc_specialty(doc)

    for sp in PCP_SPECIALTIES:
        if sp in specialty_text:
            return "pcp"

    for sp in SPECIALIST_SPECIALTIES:
        if sp in specialty_text:
            return sp

    type_text = ""
    doc_type = doc.get("type", {})
    for coding in doc_type.get("coding", []):
        code = coding.get("code", "")
        display = coding.get("display", "").lower()
        # LOINC code for H&P is 34117-2, progress note 11506-3
        if code in ("34117-2", "11506-3", "34109-9", "18842-5"):
            type_text += " " + display

    return "other"


def parse_documents(documents_bundle: dict) -> dict:
    """
    Parse DocumentReference bundle and return structured note data.

    Returns:
        {
            "pcp_notes": [{"text": ..., "date": ..., "type": ...}],
            "specialist_notes": {"cardiology": [...], "neurology": [...], ...},
            "other_notes": [...]
        }
    """
    result = {
        "pcp_notes": [],
        "specialist_notes": {},
        "other_notes": [],
    }

    entries = documents_bundle.get("entry", [])
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)

    for entry in entries:
        doc = entry.get("resource", {})
        if doc.get("resourceType") != "DocumentReference":
            continue
        if doc.get("status") == "entered-in-error":
            continue

        text = _extract_text_from_doc(doc)
        if not text:
            continue

        doc_date = _get_doc_date(doc)
        note_type = _classify_note(doc)

        authors = doc.get("author", [])
        doctor_name = authors[0].get("display") if authors else None

        note_obj = {
            "text": text,
            "date": doc_date.isoformat() if doc_date else None,
            "note_type": note_type,
            "doctor_name": doctor_name,
        }

        if note_type == "pcp":
            result["pcp_notes"].append(note_obj)
        elif note_type in SPECIALIST_SPECIALTIES:
            if note_type not in result["specialist_notes"]:
                result["specialist_notes"][note_type] = []
            if doc_date and doc_date >= six_months_ago:
                result["specialist_notes"][note_type].append(note_obj)
        else:
            result["other_notes"].append(note_obj)

    # Sort PCP notes by date, most recent first
    result["pcp_notes"].sort(key=lambda n: n["date"] or "", reverse=True)
    for sp in result["specialist_notes"]:
        result["specialist_notes"][sp].sort(key=lambda n: n["date"] or "", reverse=True)

    return result


def get_most_recent_pcp_text(parsed_notes: dict) -> str | None:
    pcp = parsed_notes.get("pcp_notes", [])
    if pcp:
        return pcp[0]["text"]
    other = parsed_notes.get("other_notes", [])
    if other:
        return other[0]["text"]
    return None


def get_all_note_text(parsed_notes: dict) -> str:
    """Combine all note text for broad keyword scanning."""
    parts = []
    for note in parsed_notes.get("pcp_notes", []):
        parts.append(note["text"])
    for sp_notes in parsed_notes.get("specialist_notes", {}).values():
        for note in sp_notes:
            parts.append(note["text"])
    for note in parsed_notes.get("other_notes", []):
        parts.append(note["text"])
    return " ".join(parts).lower()
