"""
Medication reconciliation engine.
Maps generic and brand names to drug classes and risk flags.
Returns flagged medications and determines dual therapy.
"""

import json
import re
from pathlib import Path

from clearpath.models.clinical import Medication


_MED_DATA: dict | None = None


def _load_med_data() -> dict:
    global _MED_DATA
    if _MED_DATA is None:
        path = Path(__file__).parent.parent / "data" / "medications.json"
        with open(path) as f:
            _MED_DATA = json.load(f)
    return _MED_DATA


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+\d+\s*(mg|mcg|ml|iu|units?|tab|cap|tablet|capsule|injection|solution).*", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def classify_medications(raw_names: list[str]) -> list[Medication]:
    """
    Takes a list of raw medication name strings (from FHIR).
    Returns list of Medication objects with drug class and flag populated.
    """
    med_data = _load_med_data()
    classified = []
    seen_flags = set()

    for raw_name in raw_names:
        normalized = _normalize_name(raw_name)
        matched = False

        for drug_class_key, drug_class_info in med_data.items():
            for known_name in drug_class_info["names"]:
                if known_name in normalized or normalized in known_name:
                    flag = drug_class_info["flag"]
                    tier = drug_class_info["tier"]
                    specialty = drug_class_info.get("specialty")

                    classified.append(Medication(
                        name=raw_name,
                        drug_class=drug_class_info["class"],
                        flag=flag,
                        tier=tier,
                        specialty=specialty,
                    ))
                    seen_flags.add(flag)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            classified.append(Medication(name=raw_name))

    return classified


def has_anticoagulant(medications: list[Medication]) -> bool:
    return any(m.flag == "active_anticoagulation" for m in medications)


def has_antiplatelet(medications: list[Medication]) -> bool:
    return any(m.flag == "antiplatelet_therapy" for m in medications)


def has_insulin(medications: list[Medication]) -> bool:
    return any(m.flag == "insulin_dependent_diabetes" for m in medications)


def is_dual_antiplatelet_or_combination(medications: list[Medication]) -> bool:
    anticoag_count = sum(1 for m in medications if m.flag == "active_anticoagulation")
    antiplatelet_count = sum(1 for m in medications if m.flag == "antiplatelet_therapy")
    return anticoag_count + antiplatelet_count >= 2
