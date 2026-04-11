"""
Clinical data models for ClearPath.
PatientSnapshot is the normalized internal representation.
ClearanceOutput is the canonical structured response.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Disposition(str, Enum):
    NO_CLEARANCE_NEEDED = "no_clearance_needed"
    CLEARANCE_RECOMMENDED = "clearance_recommended"
    SPECIALIST_REQUIRED = "specialist_required"
    ANESTHESIA_REVIEW_REQUIRED = "anesthesia_review_required"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SpecialistFinding(BaseModel):
    specialty: str
    last_visit_days_ago: int | None = None
    status: str
    summary: str
    doctor_name: str | None = None


class Medication(BaseModel):
    name: str
    drug_class: str | None = None
    flag: str | None = None
    tier: int | None = None
    specialty: str | None = None


class Condition(BaseModel):
    display: str
    icd_code: str | None = None
    onset_date: str | None = None
    is_chronic: bool = False


class VitalSigns(BaseModel):
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    heart_rate: int | None = None
    bmi: float | None = None
    o2_saturation: float | None = None
    recorded_date: str | None = None


class LabResult(BaseModel):
    name: str
    value: float | None = None
    unit: str | None = None
    date: str | None = None
    abnormal: bool = False


class PatientSnapshot(BaseModel):
    patient_id: str
    age: int | None = None
    sex: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    active_conditions: list[Condition] = Field(default_factory=list)
    active_medications: list[Medication] = Field(default_factory=list)
    recent_vitals: VitalSigns | None = None
    recent_labs: list[LabResult] = Field(default_factory=list)
    pcp_note_summary: str | None = None
    pcp_note_raw: str | None = None
    specialist_notes: list[dict[str, Any]] = Field(default_factory=list)
    recent_procedures: list[str] = Field(default_factory=list)
    known_implants: list[str] = Field(default_factory=list)
    days_since_pcp_visit: int | None = None
    days_since_any_lab: int | None = None
    chronic_condition_count: int = 0
    medication_count: int = 0
    extraction_warnings: list[str] = Field(default_factory=list)


class TriggerResult(BaseModel):
    trigger_id: str
    label: str
    tier: int
    specialties: list[str] = Field(default_factory=list)
    evidence: str | None = None


class ScoreResult(BaseModel):
    total_score: int
    rcri_score: int
    rcri_max: int = 6
    tier1_triggers: list[TriggerResult] = Field(default_factory=list)
    tier2_factors: list[TriggerResult] = Field(default_factory=list)
    score_breakdown: dict[str, int] = Field(default_factory=dict)


class ClearanceOutput(BaseModel):
    disposition: Disposition
    risk_level: RiskLevel
    risk_score: int
    rcri_score: int
    confidence: float
    recommended_specialties: list[str] = Field(default_factory=list)
    triggering_factors: list[str] = Field(default_factory=list)
    active_medications: list[str] = Field(default_factory=list)
    clinical_summary: str
    recommended_next_steps: list[str] = Field(default_factory=list)
    specialist_findings: list[SpecialistFinding] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    office_rule_matches: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"
    model_version: str = "clearpath-v1"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z")

    def to_markdown(self) -> str:
        disposition_label = self.disposition.value.replace("_", " ").title()
        if self.recommended_specialties and self.disposition.value in (
            "specialist_required", "anesthesia_review_required", "clearance_recommended"
        ):
            specs = ", ".join(s.title() for s in self.recommended_specialties)
            disposition_label += f": {specs}"

        lines = [
            f"## ClearPath Pre-Op Clearance Assessment",
            f"",
            f"**Disposition:** {disposition_label}",
            f"**Risk Level:** {self.risk_level.value.upper()}",
            f"**Confidence:** {int(self.confidence * 100)}%",
            f"",
            f"### Clinical Summary",
            self.clinical_summary,
        ]

        if self.active_medications:
            if len(self.active_medications) == 1:
                lines += ["", f"**Active Medications:** {self.active_medications[0]}"]
            else:
                meds_list = "\n".join(f"- {m}" for m in self.active_medications)
                lines += [
                    "",
                    f"<details><summary>Active Medications ({len(self.active_medications)})</summary>\n\n{meds_list}\n\n</details>"
                ]

        if self.triggering_factors:
            lines += ["", "### Triggering Factors"]
            for f in self.triggering_factors:
                lines.append(f"- {f}")

        if self.recommended_specialties:
            lines += ["", "### Recommended Specialties"]
            findings_by_specialty = {sf.specialty.lower(): sf for sf in self.specialist_findings}
            for s in self.recommended_specialties:
                sf = findings_by_specialty.get(s.lower())
                if sf and sf.doctor_name:
                    days = f", last seen {sf.last_visit_days_ago} days ago" if sf.last_visit_days_ago else ""
                    lines.append(f"- **{s.title()}** — {sf.doctor_name}{days}")
                elif sf:
                    days = f" (last seen {sf.last_visit_days_ago} days ago)" if sf.last_visit_days_ago else ""
                    lines.append(f"- **{s.title()}**{days}")
                else:
                    lines.append(f"- **{s.title()}** — no prior specialist on file")

        if self.specialist_findings:
            lines += ["", "### Specialist History"]
            for sf in self.specialist_findings:
                doctor = f"{sf.doctor_name} — " if sf.doctor_name else ""
                lines.append(f"- **{sf.specialty.title()}**: {doctor}{sf.summary} ({sf.status})")

        if self.recommended_next_steps:
            lines += ["", "### Recommended Next Steps"]
            for i, step in enumerate(self.recommended_next_steps, 1):
                lines.append(f"{i}. {step}")

        if self.missing_information:
            lines += ["", "### Missing Information"]
            for m in self.missing_information:
                lines.append(f"- {m}")

        lines += [
            "",
            "---",
            f"*ClearPath v{self.schema_version} | {self.generated_at[:10]} | For clinician review only*"
        ]
        return "\n".join(lines)
