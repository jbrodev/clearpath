# ClearPath sample patients

Six synthetic FHIR R4 Bundles for testing ClearPath through the Prompt Opinion platform. Each exercises a different point on the disposition / risk spectrum.

Upload via Prompt Opinion's patient import (or hand-enter the data from the bundle's PCP note if no FHIR import path exists).

| File | Patient | Age/Sex | Clinical picture | Disposition | Why it's interesting |
|---|---|---|---|---|---|
| `sample_patient_sarah_bennett.fhir.json` | Sarah Bennett | 34F | Healthy, no chronic conditions, no daily meds, wisdom-tooth extraction | **NO_CLEARANCE_NEEDED** | Demonstrates the clean-bill-of-health path |
| `sample_patient_eleanor_park.fhir.json` | Eleanor Park | 62F | Well-controlled HTN on lisinopril alone, cataract surgery | **NO_CLEARANCE_NEEDED** | Borderline: older patient on a daily med but no triggers |
| `sample_patient_marcus_thompson.fhir.json` | Marcus Thompson | 58M | Uncontrolled HTN (162/98), prediabetes, smoker, hernia repair | **NO_CLEARANCE_NEEDED** (engine under-calls) | Clinically borderline — good case to show in the demo as "the LLM follow-up catches what the rules engine misses" |
| `sample_patient_david_okafor.fhir.json` | David Okafor | 68M | AFib on rivaroxaban, T2DM, HTN, hyperlipidemia, colonoscopy | **SPECIALIST_REQUIRED** | Multi-specialty, anticoagulation hold question, prescriber-scope reasoning |
| `sample_patient_linda_rivera.fhir.json` | Linda Rivera | 71F | Recent NSTEMI w/ DES on DAPT, O2-dependent COPD, severe OSA, CKD, hip replacement | **SPECIALIST_REQUIRED** (cardiology, hematology, pulmonology) | Highest-acuity case — DAPT timing + OSA + COPD |
| `sample_patient_robert_hale.fhir.json` | Robert Hale | 78M | Empty chart, no conditions, no meds, no recent notes | **INSUFFICIENT_INFORMATION** | Demonstrates the "we can't assess yet" path |

## Recommended demo order

1. **Sarah** — show the easy case. Quick clean assessment.
2. **Robert** — show how ClearPath refuses to assess without data (safety property).
3. **David** — show the specialist-referral path; ask follow-ups about rivaroxaban hold and prescriber scope.
4. **Linda** — show the high-acuity case with three specialty referrals; ask about DAPT timing.
5. **Marcus** (optional) — interesting because the deterministic engine says clear, but the LLM follow-up will surface the BP concern when you ask.
6. **Eleanor** (optional) — borderline daily-meds case.

## Good follow-up questions per patient

After the initial clearance lands, try these — they hit streaming + web_search + the prescriber-scope reasoning:

- **Sarah** — *"Any concerns for IV sedation in a healthy adult?"*
- **Eleanor** — *"Should she take her lisinopril the morning of cataract surgery?"* (ACE inhibitor day-of management — ACC/AHA + ASA guidance)
- **Marcus** — *"What blood pressure threshold should defer this procedure?"* (ASA Class III / Stage 2 HTN; tests whether the LLM surfaces what the rules engine missed)
- **David** — *"When should we hold his rivaroxaban before the colonoscopy?"* | *"Does Dr. Chen manage all his medications?"* (prescriber-scope test)
- **Linda** — *"Why can't we proceed with the hip replacement now?"* | *"What's the bleed risk of holding her DAPT for this surgery?"*
- **Robert** — *"What data do you need before I can run a clearance on him?"*

## Notes on engine behavior

ClearPath's deterministic rule engine is intentionally conservative — it only escalates to a higher disposition when a Tier-1 trigger fires (anticoagulation, cardiac event, oxygen dependence, etc.). Borderline patients like Marcus correctly fall to `NO_CLEARANCE_NEEDED` because they have no Tier-1 triggers. The follow-up LLM reasoning layer is where soft clinical concerns (uncontrolled BP, A1c trends, smoking status) get surfaced — that's a feature, not a bug.

## Regenerating

Edit `_generate.py` and re-run:
```
python samples/_generate.py
```
