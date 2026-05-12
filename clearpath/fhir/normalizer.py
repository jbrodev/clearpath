"""
Converts raw FHIR bundles + parsed notes into a PatientSnapshot.
This is the single function that translates messy FHIR into structured
clinical state that the trigger engine and LLM can use.
"""

import re
from datetime import datetime, timedelta, timezone

from clearpath.fhir.note_parser import parse_documents, get_most_recent_pcp_text, get_all_note_text
from clearpath.models.clinical import (
    PatientSnapshot, Condition, Medication, VitalSigns, LabResult, SpecialistFinding
)


def _get_entries(bundle: dict) -> list:
    if not bundle or "_error" in bundle:
        return []
    if bundle.get("resourceType") == "Bundle":
        return [e.get("resource", {}) for e in bundle.get("entry", []) if e.get("resource")]
    if bundle.get("resourceType"):
        return [bundle]
    return []


def _extract_patient(patient_resource: dict) -> dict:
    result = {"age": None, "sex": None, "first_name": None, "last_name": None, "birth_date": None}
    if not patient_resource or patient_resource.get("resourceType") != "Patient":
        return result

    dob = patient_resource.get("birthDate")
    if dob:
        result["birth_date"] = dob
        try:
            birth = datetime.fromisoformat(dob[:10]).replace(tzinfo=timezone.utc)
            today = datetime.now(timezone.utc)
            result["age"] = (today - birth).days // 365
        except Exception:
            pass

    gender = patient_resource.get("gender")
    if gender:
        result["sex"] = gender

    name_list = patient_resource.get("name", [])
    if name_list:
        name_entry = name_list[0]
        result["last_name"] = name_entry.get("family")
        given = name_entry.get("given") or []
        result["first_name"] = given[0] if given else None

    return result


def _extract_conditions(conditions_bundle: dict) -> list[Condition]:
    conditions = []
    for resource in _get_entries(conditions_bundle):
        if resource.get("resourceType") != "Condition":
            continue
        code = resource.get("code", {})
        display = ""
        icd = None
        for coding in code.get("coding", []):
            if not display:
                display = coding.get("display", "")
            system = coding.get("system", "")
            if "icd" in system.lower() or "2.16.840.1.113883.6" in system:
                icd = coding.get("code")
        if not display:
            display = code.get("text", "unknown condition")

        onset = resource.get("onsetDateTime") or resource.get("onsetString")
        conditions.append(Condition(
            display=display,
            icd_code=icd,
            onset_date=onset[:10] if onset else None,
            is_chronic=True
        ))
    return conditions


_DOSAGE_RE = re.compile(r'\d+\s*(mg|mcg|ml|iu|units?)', re.IGNORECASE)
_BASE_STRIP_RE = re.compile(r'\s+\d+.*')


def _best_med_name(resource: dict) -> str | None:
    """Pick the single best medication name from a FHIR resource, preferring the one with dosage."""
    med = resource.get("medicationCodeableConcept") or resource.get("medication", {})
    if not isinstance(med, dict):
        return None
    candidates = []
    for coding in med.get("coding", []):
        display = coding.get("display", "").strip()
        if display:
            candidates.append(display)
    text = med.get("text", "").strip()
    if text:
        candidates.append(text)
    if not candidates:
        return None
    with_dosage = [c for c in candidates if _DOSAGE_RE.search(c)]
    return (with_dosage[0] if with_dosage else candidates[0]).lower()


def _extract_medications(meds_bundle: dict) -> list[str]:
    """Returns deduplicated medication name list (one entry per drug, with dosage preferred)."""
    raw = []
    for resource in _get_entries(meds_bundle):
        if resource.get("resourceType") not in ("MedicationRequest", "MedicationStatement"):
            continue
        name = _best_med_name(resource)
        if name:
            raw.append(name)
    # Deduplicate across resources by base name, keeping the most specific (longest/with dosage)
    seen: dict[str, str] = {}
    for name in raw:
        base = _BASE_STRIP_RE.sub("", name).strip()
        existing = seen.get(base)
        if not existing or len(name) > len(existing):
            seen[base] = name
    return list(seen.values())


def _extract_vitals(vitals_bundle: dict) -> VitalSigns:
    vitals = VitalSigns()
    systolic_vals = []
    diastolic_vals = []

    for resource in _get_entries(vitals_bundle):
        if resource.get("resourceType") != "Observation":
            continue
        code = resource.get("code", {})
        loinc = ""
        for coding in code.get("coding", []):
            loinc = coding.get("code", "")

        value_qty = resource.get("valueQuantity", {})
        value = value_qty.get("value")

        components = resource.get("component", [])
        if components:
            for comp in components:
                comp_code = comp.get("code", {})
                comp_loinc = ""
                for c in comp_code.get("coding", []):
                    comp_loinc = c.get("code", "")
                comp_val = comp.get("valueQuantity", {}).get("value")
                if comp_loinc == "8480-6" and comp_val:
                    systolic_vals.append(int(comp_val))
                elif comp_loinc == "8462-4" and comp_val:
                    diastolic_vals.append(int(comp_val))

        if loinc == "8480-6" and value:
            systolic_vals.append(int(value))
        elif loinc == "8462-4" and value:
            diastolic_vals.append(int(value))
        elif loinc == "8867-4" and value:
            vitals.heart_rate = int(value)
        elif loinc == "39156-5" and value:
            vitals.bmi = float(value)
        elif loinc == "59408-5" and value:
            vitals.o2_saturation = float(value)

        date_str = resource.get("effectiveDateTime")
        if date_str and not vitals.recorded_date:
            vitals.recorded_date = date_str[:10]

    if systolic_vals:
        vitals.systolic_bp = max(systolic_vals)
    if diastolic_vals:
        vitals.diastolic_bp = max(diastolic_vals)

    return vitals


def _extract_labs(labs_bundle: dict) -> list[LabResult]:
    labs = []
    seen = set()
    for resource in _get_entries(labs_bundle):
        if resource.get("resourceType") != "Observation":
            continue
        code = resource.get("code", {})
        display = code.get("text", "")
        for coding in code.get("coding", []):
            if not display:
                display = coding.get("display", "")

        if not display or display in seen:
            continue
        seen.add(display)

        value_qty = resource.get("valueQuantity", {})
        value = value_qty.get("value")
        unit = value_qty.get("unit")
        date_str = resource.get("effectiveDateTime")
        interp = resource.get("interpretation", [{}])
        abnormal = False
        if interp:
            for i in interp:
                for c in i.get("coding", []):
                    code_val = c.get("code", "")
                    if code_val in ("H", "L", "HH", "LL", "A", "AA", "C"):
                        abnormal = True

        labs.append(LabResult(
            name=display,
            value=float(value) if value is not None else None,
            unit=unit,
            date=date_str[:10] if date_str else None,
            abnormal=abnormal
        ))
    return labs


def _extract_procedures(procedures_bundle: dict) -> list[str]:
    procs = []
    for resource in _get_entries(procedures_bundle):
        if resource.get("resourceType") != "Procedure":
            continue
        code = resource.get("code", {})
        display = code.get("text", "")
        for coding in code.get("coding", []):
            if not display:
                display = coding.get("display", "")
        if display:
            procs.append(display)
    return procs


def _days_since_last_pcp_visit(encounters_bundle: dict) -> int | None:
    pcp_keywords = {"family medicine", "internal medicine", "primary care", "general practice"}
    latest_date = None

    for resource in _get_entries(encounters_bundle):
        if resource.get("resourceType") != "Encounter":
            continue
        type_list = resource.get("type", [])
        service_type = resource.get("serviceType", {})
        text = ""
        for t in type_list:
            for c in t.get("coding", []):
                text += " " + c.get("display", "").lower()
        for c in service_type.get("coding", []):
            text += " " + c.get("display", "").lower()

        is_pcp = any(kw in text for kw in pcp_keywords) or not text.strip()
        if not is_pcp:
            continue

        period = resource.get("period", {})
        date_str = period.get("end") or period.get("start")
        if not date_str:
            continue
        try:
            visit_date = datetime.fromisoformat(date_str[:10]).replace(tzinfo=timezone.utc)
            if latest_date is None or visit_date > latest_date:
                latest_date = visit_date
        except Exception:
            pass

    if latest_date:
        return (datetime.now(timezone.utc) - latest_date).days
    return None


def _days_since_any_lab(labs: list[LabResult]) -> int | None:
    latest = None
    for lab in labs:
        if lab.date:
            try:
                d = datetime.fromisoformat(lab.date).replace(tzinfo=timezone.utc)
                if latest is None or d > latest:
                    latest = d
            except Exception:
                pass
    if latest:
        return (datetime.now(timezone.utc) - latest).days
    return None


def _find_implants(conditions: list[Condition], all_note_text: str) -> list[str]:
    implant_keywords = [
        "pacemaker", "icd", "implantable cardioverter",
        "cardiac resynchronization", "crt-d", "crt-p",
        "cochlear implant", "deep brain stimulator", "dbs",
        "ventricular assist device", "vad",
    ]
    found = []
    combined = all_note_text + " ".join(c.display.lower() for c in conditions)
    for kw in implant_keywords:
        if kw in combined and kw not in found:
            found.append(kw)
    return found


def build_snapshot(fhir_data: dict) -> PatientSnapshot:
    """
    Convert raw FHIR bundles into a normalized PatientSnapshot.
    fhir_data keys: patient, conditions, medications, procedures,
                    documents, vitals, labs, encounters, allergies
    """
    warnings = []

    patient_raw = fhir_data.get("patient", {})
    patient_info = _extract_patient(patient_raw)

    conditions = _extract_conditions(fhir_data.get("conditions", {}))
    if not conditions:
        warnings.append("No active conditions found in FHIR")

    raw_med_names = _extract_medications(fhir_data.get("medications", {}))
    if not raw_med_names:
        warnings.append("No active medications found in FHIR")

    vitals = _extract_vitals(fhir_data.get("vitals", {}))
    if not vitals.systolic_bp:
        warnings.append("No recent blood pressure readings found")

    labs = _extract_labs(fhir_data.get("labs", {}))
    if not labs:
        warnings.append("No recent lab results found")

    procedures = _extract_procedures(fhir_data.get("procedures", {}))

    parsed_notes = parse_documents(fhir_data.get("documents", {}))
    pcp_note_raw = get_most_recent_pcp_text(parsed_notes)
    all_note_text = get_all_note_text(parsed_notes)
    pcp_notes_list = parsed_notes.get("pcp_notes", [])
    pcp_doctor_name = pcp_notes_list[0].get("doctor_name") if pcp_notes_list else None

    if not pcp_note_raw:
        warnings.append("No PCP or primary care notes found")

    days_pcp = _days_since_last_pcp_visit(fhir_data.get("encounters", {}))
    days_labs = _days_since_any_lab(labs)

    implants = _find_implants(conditions, all_note_text)
    chronic_count = len([c for c in conditions if c.is_chronic])

    # Build structured specialist findings from parsed notes
    specialist_note_data = parsed_notes.get("specialist_notes", {})

    return PatientSnapshot(
        patient_id=patient_raw.get("id", "unknown"),
        age=patient_info["age"],
        sex=patient_info["sex"],
        first_name=patient_info["first_name"],
        last_name=patient_info["last_name"],
        birth_date=patient_info.get("birth_date"),
        active_conditions=conditions,
        active_medications=[Medication(name=n) for n in raw_med_names],
        recent_vitals=vitals,
        recent_labs=labs,
        pcp_note_raw=pcp_note_raw,
        pcp_doctor_name=pcp_doctor_name,
        specialist_notes=[
            {"specialty": sp, "notes": notes}
            for sp, notes in specialist_note_data.items()
        ],
        recent_procedures=procedures,
        known_implants=implants,
        days_since_pcp_visit=days_pcp,
        days_since_any_lab=days_labs,
        chronic_condition_count=chronic_count,
        medication_count=len(raw_med_names),
        extraction_warnings=warnings,
    )
