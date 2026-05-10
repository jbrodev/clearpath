"""
One-off generator for synthetic FHIR Bundle test patients (PromptOpinion upload).
Run: python samples/_generate.py
Produces sample_patient_*.fhir.json files in this folder.

Bundle shape: transaction Bundle with POST entries and urn:uuid fullUrls.
PromptOpinion's FHIR server does not support updateCreate (PUT against an
id that doesn't yet exist), so we POST each resource and let the server
assign ids. Inter-resource references use urn:uuid: so the Patient and
all related resources stitch together in one atomic transaction.

Re-uploading the same file creates duplicates (no conditional-create), so
delete the previous patient in Po if you re-test.
"""

import base64
import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def make_patient(family: str, given: str, gender: str, birth: str) -> dict:
    return {
        "resourceType": "Patient",
        "name": [{"family": family, "given": [given]}],
        "gender": gender,
        "birthDate": birth,
    }


def make_condition(patient_ref: str, icd: str, display: str, text: str, onset: str) -> dict:
    return {
        "resourceType": "Condition",
        "subject": {"reference": patient_ref},
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "code": {
            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": icd, "display": display}],
            "text": text,
        },
        "onsetDateTime": onset,
    }


def make_med(patient_ref: str, rxcui: str, display: str, text: str, authored: str) -> dict:
    return {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "subject": {"reference": patient_ref},
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": rxcui, "display": display}],
            "text": text,
        },
        "authoredOn": authored,
    }


def make_bp(patient_ref: str, sys_val: int, dia_val: int, date: str) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel"}]},
        "subject": {"reference": patient_ref},
        "effectiveDateTime": date,
        "component": [
            {"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]}, "valueQuantity": {"value": sys_val, "unit": "mmHg"}},
            {"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4"}]}, "valueQuantity": {"value": dia_val, "unit": "mmHg"}},
        ],
    }


def make_vital(patient_ref: str, loinc: str, display: str, value, unit: str, date: str) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]},
        "subject": {"reference": patient_ref},
        "effectiveDateTime": date,
        "valueQuantity": {"value": value, "unit": unit},
    }


def make_lab(patient_ref: str, loinc: str, display: str, value, unit: str, date: str, abnormal: bool = False) -> dict:
    obs = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]},
        "subject": {"reference": patient_ref},
        "effectiveDateTime": date,
        "valueQuantity": {"value": value, "unit": unit},
    }
    if abnormal:
        obs["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H"}]}]
    return obs


def make_note(patient_ref: str, category: str, loinc: str, loinc_display: str, date: str, author_name: str, text: str) -> dict:
    return {
        "resourceType": "DocumentReference",
        "status": "current",
        "type": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": loinc_display}]},
        "category": [{"text": category}],
        "subject": {"reference": patient_ref},
        "date": date,
        "author": [{"display": author_name}],
        "content": [{"attachment": {"contentType": "text/plain", "data": b64(text)}}],
    }


def make_encounter(patient_ref: str, snomed_code: str, snomed_display: str, start: str, end: str) -> dict:
    return {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
        "type": [{"coding": [{"system": "http://snomed.info/sct", "code": snomed_code, "display": snomed_display}]}],
        "subject": {"reference": patient_ref},
        "period": {"start": start, "end": end},
    }


def transaction_entry(resource: dict, full_url: str) -> dict:
    return {
        "fullUrl": full_url,
        "resource": resource,
        "request": {"method": "POST", "url": resource["resourceType"]},
    }


def build_bundle(builder):
    """builder(patient_ref) -> (patient_resource, [resource, ...])"""
    patient_url = urn()
    patient_resource, others = builder(patient_url)
    entries = [transaction_entry(patient_resource, patient_url)]
    for r in others:
        entries.append(transaction_entry(r, urn()))
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def write(filename: str, content: dict) -> None:
    path = OUT / filename
    path.write_text(json.dumps(content, indent=2))
    print(f"  wrote {path.name} ({len(content['entry'])} resources)")


# ------------------------------------------------------------------
# Patient 1 — Sarah Bennett, 34F. Healthy. Expected: NO_CLEARANCE_NEEDED
# ------------------------------------------------------------------
def sarah(p):
    pcp_text = (
        "Healthy 34-year-old woman. Annual physical. No chronic conditions, no daily "
        "medications other than a multivitamin. Non-smoker. Exercises 4 times per week. "
        "BMI normal. Vitals normal. Routine bloodwork normal. Patient scheduling wisdom "
        "tooth extraction with sedation - requested pre-op clearance per OMS office "
        "policy. No anesthesia concerns from primary care perspective."
    )
    return make_patient("Bennett", "Sarah", "female", "1992-08-04"), [
        make_bp(p, 118, 72, "2026-04-30"),
        make_vital(p, "8867-4", "Heart rate", 68, "bpm", "2026-04-30"),
        make_vital(p, "39156-5", "Body mass index", 22.1, "kg/m2", "2026-04-30"),
        make_lab(p, "718-7", "Hemoglobin", 13.6, "g/dL", "2026-04-30"),
        make_lab(p, "2160-0", "Creatinine", 0.8, "mg/dL", "2026-04-30"),
        make_note(p, "family medicine", "11506-3", "Progress note", "2026-04-30", "Dr. Lisa Marquez, Family Medicine", pcp_text),
        make_encounter(p, "185349003", "Family medicine consultation", "2026-04-30T10:00:00Z", "2026-04-30T10:20:00Z"),
    ]


# ------------------------------------------------------------------
# Patient 2 — Eleanor Park, 62F. Well-controlled HTN, single daily med.
# Expected: NO_CLEARANCE_NEEDED (borderline)
# ------------------------------------------------------------------
def eleanor(p):
    pcp_text = (
        "62-year-old woman, hypertension well-controlled on monotherapy x 6 years. "
        "No other chronic conditions. Non-smoker. Walks daily. BP today 124/78, has been "
        "consistently below 130/80. Annual labs unremarkable. Scheduling cataract surgery "
        "left eye - low-risk procedure. Continue lisinopril through procedure per anesthesia preference."
    )
    return make_patient("Park", "Eleanor", "female", "1964-02-19"), [
        make_condition(p, "I10", "Essential hypertension", "Hypertension", "2020-05-10"),
        make_med(p, "314076", "lisinopril 10 MG Oral Tablet", "lisinopril 10mg daily", "2024-06-12"),
        make_bp(p, 124, 78, "2026-05-01"),
        make_vital(p, "8867-4", "Heart rate", 72, "bpm", "2026-05-01"),
        make_vital(p, "39156-5", "Body mass index", 24.8, "kg/m2", "2026-05-01"),
        make_lab(p, "2160-0", "Creatinine", 0.9, "mg/dL", "2026-04-25"),
        make_lab(p, "2823-3", "Potassium", 4.1, "mmol/L", "2026-04-25"),
        make_note(p, "family medicine", "11506-3", "Progress note", "2026-05-01", "Dr. Henry Williams, Family Medicine", pcp_text),
        make_encounter(p, "185349003", "Family medicine consultation", "2026-05-01T14:00:00Z", "2026-05-01T14:25:00Z"),
    ]


# ------------------------------------------------------------------
# Patient 3 — Marcus Thompson, 58M. Uncontrolled HTN, prediabetes, smoker.
# ------------------------------------------------------------------
def marcus(p):
    pcp_text = (
        "58-year-old man with hypertension uncontrolled on two agents (amlodipine + HCTZ). "
        "BP today 162/98, last 3 readings all >150/90. Prediabetes per A1c 6.2%. Continues "
        "to smoke 1 pack per day, 30-pack-year history. BMI 29. Counseled on BP and smoking "
        "again - patient declines specialist referral, will trial third antihypertensive "
        "(losartan added). Planning open inguinal hernia repair next month. BP needs to be "
        "below 160/100 day-of-surgery; if not, will need to defer or get anesthesia input."
    )
    return make_patient("Thompson", "Marcus", "male", "1968-11-23"), [
        make_condition(p, "I10", "Essential hypertension", "Hypertension (uncontrolled)", "2017-09-14"),
        make_condition(p, "R73.03", "Prediabetes", "Prediabetes", "2023-02-08"),
        make_condition(p, "F17.210", "Nicotine dependence, cigarettes", "Tobacco use disorder", "2010-01-01"),
        make_med(p, "197361", "amlodipine 10 MG Oral Tablet", "amlodipine 10mg daily", "2025-08-15"),
        make_med(p, "310798", "hydrochlorothiazide 25 MG Oral Tablet", "hydrochlorothiazide 25mg daily", "2025-08-15"),
        make_med(p, "979467", "losartan 50 MG Oral Tablet", "losartan 50mg daily (newly added)", "2026-04-28"),
        make_bp(p, 162, 98, "2026-04-28"),
        make_vital(p, "8867-4", "Heart rate", 84, "bpm", "2026-04-28"),
        make_vital(p, "39156-5", "Body mass index", 29.4, "kg/m2", "2026-04-28"),
        make_lab(p, "4548-4", "Hemoglobin A1c", 6.2, "%", "2026-04-20"),
        make_lab(p, "2160-0", "Creatinine", 1.0, "mg/dL", "2026-04-20"),
        make_lab(p, "13457-7", "LDL cholesterol", 138, "mg/dL", "2026-04-20", abnormal=True),
        make_note(p, "family medicine", "11506-3", "Progress note", "2026-04-28", "Dr. Henry Williams, Family Medicine", pcp_text),
        make_encounter(p, "185349003", "Family medicine consultation", "2026-04-28T11:00:00Z", "2026-04-28T11:30:00Z"),
    ]


# ------------------------------------------------------------------
# Patient 4 — David Okafor, 68M. AFib on rivaroxaban, T2DM, HTN, HLD.
# Expected: SPECIALIST_REQUIRED, high risk
# ------------------------------------------------------------------
def david(p):
    pcp_text = (
        "Mr. Okafor is a 68-year-old man with paroxysmal atrial fibrillation, hypertension, "
        "type 2 diabetes, and hyperlipidemia. Anticoagulated on rivaroxaban. Metoprolol for "
        "rate control, lisinopril for BP, metformin for DM, atorvastatin for hyperlipidemia. "
        "BP 142/88 today. A1c 7.8. Cr 1.3. Planning screening colonoscopy in 6 weeks - "
        "requesting pre-op clearance assessment. No recent chest pain. No dyspnea. Functional "
        "status good, >4 METs. Cardiology (Dr. Karen Chen) following for AF management."
    )
    cards_text = (
        "Mr. Okafor followed for paroxysmal atrial fibrillation. On rivaroxaban 20mg daily "
        "for stroke prevention (CHA2DS2-VASc = 4). Metoprolol succinate for rate control, "
        "tolerating well. Recent echo shows preserved LVEF 60%. No valvular disease. No "
        "ischemic heart disease. Reassess in 6 months."
    )
    return make_patient("Okafor", "David", "male", "1957-04-12"), [
        make_condition(p, "I48.91", "Atrial fibrillation, unspecified", "Atrial fibrillation", "2022-08-15"),
        make_condition(p, "I10", "Essential hypertension", "Hypertension", "2018-03-04"),
        make_condition(p, "E11.9", "Type 2 diabetes mellitus without complications", "Type 2 diabetes mellitus", "2015-06-20"),
        make_condition(p, "E78.5", "Hyperlipidemia, unspecified", "Hyperlipidemia", "2016-01-12"),
        make_med(p, "1114195", "rivaroxaban 20 MG Oral Tablet", "rivaroxaban 20mg daily", "2024-11-04"),
        make_med(p, "866427", "metoprolol succinate 50 MG Extended Release Oral Tablet", "metoprolol succinate 50mg daily", "2024-11-04"),
        make_med(p, "314076", "lisinopril 20 MG Oral Tablet", "lisinopril 20mg daily", "2024-11-04"),
        make_med(p, "860975", "metformin hydrochloride 1000 MG Oral Tablet", "metformin 1000mg twice daily", "2024-11-04"),
        make_med(p, "617312", "atorvastatin 40 MG Oral Tablet", "atorvastatin 40mg nightly", "2024-11-04"),
        make_bp(p, 142, 88, "2026-04-22"),
        make_vital(p, "8867-4", "Heart rate", 78, "bpm", "2026-04-22"),
        make_vital(p, "39156-5", "Body mass index", 31.4, "kg/m2", "2026-04-22"),
        make_lab(p, "4548-4", "Hemoglobin A1c", 7.8, "%", "2026-04-15", abnormal=True),
        make_lab(p, "2160-0", "Creatinine", 1.3, "mg/dL", "2026-04-15"),
        make_lab(p, "13457-7", "LDL cholesterol", 92, "mg/dL", "2026-04-15"),
        make_note(p, "family medicine", "11506-3", "Progress note", "2026-04-22", "Dr. Amita Patel, Family Medicine", pcp_text),
        make_note(p, "cardiology", "11488-4", "Consult note", "2026-02-18", "Dr. Karen Chen, Cardiology", cards_text),
        make_encounter(p, "185349003", "Family medicine consultation", "2026-04-22T09:00:00Z", "2026-04-22T09:30:00Z"),
    ]


# ------------------------------------------------------------------
# Patient 5 — Linda Rivera, 71F. NSTEMI w/ DES on DAPT, O2-dependent COPD, severe OSA, CKD.
# Expected: SPECIALIST_REQUIRED, high/critical risk
# ------------------------------------------------------------------
def linda(p):
    pcp_text = (
        "Complex 71-year-old woman with multiple high-risk conditions: paroxysmal AFib "
        "controlled on metoprolol, NSTEMI 4 months ago with DES placement in LAD - on dual "
        "antiplatelet therapy (aspirin + clopidogrel), severe COPD requiring 2L home oxygen, "
        "severe obstructive sleep apnea on nightly CPAP, CKD stage 3 (eGFR 38), obesity (BMI 34). "
        "Cardiology and pulmonology are actively involved. Patient is being evaluated for "
        "elective right total hip replacement. Given recent NSTEMI/stent (within 6 months), "
        "DAPT requirement, oxygen-dependent COPD, severe OSA, and CKD, this patient requires "
        "BOTH cardiology and anesthesia review before any elective procedure can be considered."
    )
    cards_text = (
        "Mrs. Rivera, 4 months s/p NSTEMI with successful PCI and drug-eluting stent placement "
        "in LAD. Currently on aspirin 81mg + clopidogrel 75mg (DAPT). Plan to continue DAPT "
        "for minimum 6 months, ideally 12 months given DES. Beta blocker (metoprolol) and "
        "high-intensity statin continued. Echo last month: LVEF 50%, mild mitral regurg. "
        "Strong recommendation: do NOT interrupt DAPT for elective surgery until 6 months "
        "post-PCI at the earliest."
    )
    pulm_text = (
        "Severe COPD GOLD stage 3, FEV1 42% predicted. On home O2 at 2L. Severe OSA, AHI 38, "
        "fully compliant with CPAP at 12 cm H2O. Tobacco-free x 4 years. On tiotropium daily, "
        "albuterol rescue. Stable but high perioperative pulmonary risk. ARISCAT high. "
        "Recommend regional anesthesia if at all possible."
    )
    return make_patient("Rivera", "Linda", "female", "1955-03-08"), [
        make_condition(p, "I48.91", "Atrial fibrillation, unspecified", "Atrial fibrillation, paroxysmal", "2020-04-12"),
        make_condition(p, "I21.4", "Non-ST elevation myocardial infarction", "NSTEMI s/p DES (4 months ago)", "2026-01-15"),
        make_condition(p, "J44.9", "Chronic obstructive pulmonary disease", "COPD, oxygen-dependent", "2018-11-03"),
        make_condition(p, "G47.33", "Obstructive sleep apnea (adult)", "Severe obstructive sleep apnea on CPAP", "2019-06-21"),
        make_condition(p, "N18.3", "Chronic kidney disease, stage 3", "CKD stage 3", "2022-08-30"),
        make_condition(p, "E66.9", "Obesity, unspecified", "Obesity (BMI 34)", "2015-01-01"),
        make_med(p, "243670", "aspirin 81 MG Oral Tablet", "aspirin 81mg daily", "2026-01-16"),
        make_med(p, "309362", "clopidogrel 75 MG Oral Tablet", "clopidogrel 75mg daily", "2026-01-16"),
        make_med(p, "866427", "metoprolol succinate 50 MG Extended Release Oral Tablet", "metoprolol succinate 50mg daily", "2026-01-16"),
        make_med(p, "617318", "atorvastatin 80 MG Oral Tablet", "atorvastatin 80mg nightly", "2026-01-16"),
        make_med(p, "1543450", "tiotropium bromide 18 MCG Inhalant Capsule", "tiotropium 18mcg inhaled daily", "2023-09-04"),
        make_med(p, "329498", "albuterol 0.09 MG/ACTUAT Metered Dose Inhaler", "albuterol HFA 2 puffs PRN", "2023-09-04"),
        make_bp(p, 138, 82, "2026-04-29"),
        make_vital(p, "8867-4", "Heart rate", 76, "bpm", "2026-04-29"),
        make_vital(p, "39156-5", "Body mass index", 34.1, "kg/m2", "2026-04-29"),
        make_vital(p, "59408-5", "Oxygen saturation", 92, "%", "2026-04-29"),
        make_lab(p, "2160-0", "Creatinine", 1.6, "mg/dL", "2026-04-22", abnormal=True),
        make_lab(p, "33914-3", "Glomerular filtration rate", 38, "mL/min/1.73m2", "2026-04-22", abnormal=True),
        make_lab(p, "718-7", "Hemoglobin", 11.4, "g/dL", "2026-04-22", abnormal=True),
        make_lab(p, "777-3", "Platelets", 188, "10*3/uL", "2026-04-22"),
        make_note(p, "family medicine", "11506-3", "Progress note", "2026-04-29", "Dr. Sandra Okafor, Family Medicine", pcp_text),
        make_note(p, "cardiology", "11488-4", "Consult note", "2026-04-15", "Dr. Wei Liu, Cardiology", cards_text),
        make_note(p, "pulmonology", "11488-4", "Consult note", "2026-03-20", "Dr. Priya Sharma, Pulmonology", pulm_text),
        make_encounter(p, "185349003", "Family medicine consultation", "2026-04-29T13:00:00Z", "2026-04-29T13:40:00Z"),
    ]


# ------------------------------------------------------------------
# Patient 6 — Robert Hale, 78M. Empty chart. Expected: INSUFFICIENT_INFORMATION
# ------------------------------------------------------------------
def robert(p):
    return make_patient("Hale", "Robert", "male", "1948-07-15"), [
        make_encounter(p, "185349003", "Family medicine consultation", "2025-08-04T09:00:00Z", "2025-08-04T09:15:00Z"),
    ]


if __name__ == "__main__":
    print("Generating synthetic FHIR transaction Bundles:")
    write("sample_patient_sarah_bennett.fhir.json", build_bundle(sarah))
    write("sample_patient_eleanor_park.fhir.json", build_bundle(eleanor))
    write("sample_patient_marcus_thompson.fhir.json", build_bundle(marcus))
    write("sample_patient_david_okafor.fhir.json", build_bundle(david))
    write("sample_patient_linda_rivera.fhir.json", build_bundle(linda))
    write("sample_patient_robert_hale.fhir.json", build_bundle(robert))
    print("Done.")
