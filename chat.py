"""
ClearPath interactive agent.
Type a patient name or ask a question to get started.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Load .env first so it can set ANTHROPIC_API_KEY before any defaults
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("ANTHROPIC_API_KEY", "")

PATIENTS_DIR = Path(__file__).parent / "clearpath" / "data" / "synthetic_patients"

# Legacy letter shortcuts (backward compat)
LEGACY_KEYS = {
    "a": "patient_a_high_risk.json",
    "b": "patient_b_low_risk.json",
    "e": "patient_e_sparse.json",
}

BANNER = """
==========================================
  ClearPath  -  Pre-Op Clearance Agent
==========================================

Ask about any patient by name:

  "Is Maria Gonzalez cleared for surgery?"
  "Run a clearance check on Robert Chen"
  "What are John Doe's anesthesia risks?"

Once a patient is loaded, keep asking:

  "What does her COPD mean for recovery?"
  "Explain his blood thinner in plain terms"
  "Should she stop her metformin before the procedure?"

Type  'patients'  to see who is available.
Type  'help'      to see this message again.
Type  'quit'      to exit.
"""

# --- Patient registry ---

def build_registry() -> list[dict]:
    """Scan synthetic_patients/ and build a lookup list from FHIR patient resources."""
    registry = []
    for path in sorted(PATIENTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            p = data.get("patient", {})
            if p.get("resourceType") != "Patient":
                continue
            name_list = p.get("name", [])
            name_entry = name_list[0] if name_list else {}
            given = name_entry.get("given") or []
            first = given[0] if given else None
            last = name_entry.get("family")
            birth_date = p.get("birthDate")
            gender = p.get("gender", "unknown")
            patient_id = p.get("id", path.stem)
            registry.append({
                "file": path,
                "patient_id": patient_id,
                "first_name": first,
                "last_name": last,
                "birth_date": birth_date,
                "gender": gender,
            })
        except Exception:
            pass
    return registry


# --- DOB parsing ---

_DOB_PATTERNS = [
    # MM/DD/YYYY or M/D/YYYY
    (re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'), lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
    # YYYY-MM-DD
    (re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'), lambda m: m.group(0)),
    # MM-DD-YYYY
    (re.compile(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b'), lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
]

def _extract_dob(text: str) -> tuple[str, str | None]:
    """Return (text_with_dob_removed, iso_dob_or_None)."""
    for pattern, formatter in _DOB_PATTERNS:
        m = pattern.search(text)
        if m:
            iso = formatter(m)
            cleaned = pattern.sub("", text).replace("dob", "").replace("DOB", "").strip()
            return cleaned, iso
    return text, None


# --- Clinical keyword heuristic ---

_CLINICAL_WORDS = {
    "cleared", "clearance", "surgery", "procedure", "anesthesia", "risk",
    "medication", "medications", "condition", "conditions", "cardiac",
    "heart", "stroke", "diabetes", "hypertension", "blood", "labs",
    "recommend", "specialist", "consult", "review", "assessment",
    "pre-op", "preop", "operative", "is", "are", "what", "should",
    "does", "do", "can", "will", "how", "why", "when", "safe",
    "concern", "issue", "problem", "history", "note", "please",
}

def _looks_like_name_query(text: str) -> bool:
    """Return True if the input looks like a name search rather than a clinical question."""
    tokens = text.lower().split()
    if len(tokens) > 6:
        return False
    # If any token is a strong clinical word, treat as query
    for tok in tokens:
        clean = tok.strip(".,?!")
        if clean in _CLINICAL_WORDS:
            return False
    # Must have at least one alphabetic token (the name)
    alpha_tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
    return len(alpha_tokens) >= 1


# --- Fuzzy matching ---

def _normalize(s: str) -> str:
    return re.sub(r'[^a-z]', '', s.lower())

def find_patient_in_query(registry: list[dict], text: str) -> list[dict]:
    """Scan a clinical question for any patient name that appears in the registry."""
    norm_text = _normalize(text)
    matches = []
    for entry in registry:
        first = _normalize(entry["first_name"] or "")
        last = _normalize(entry["last_name"] or "")
        # Require at least last name to appear in the text
        if last and last in norm_text:
            # Bonus: also check first name to avoid false positives on common last names
            if first and first in norm_text:
                matches.append(entry)
            elif not first:
                matches.append(entry)
    return matches


def find_patients(registry: list[dict], name_str: str, dob: str | None) -> list[dict]:
    """Match patients by name (fuzzy) and optionally DOB (exact)."""
    parts = [p for p in name_str.split() if re.match(r'^[a-zA-Z]', p)]
    if not parts:
        return []

    norm_parts = [_normalize(p) for p in parts]

    matches = []
    for entry in registry:
        first = _normalize(entry["first_name"] or "")
        last = _normalize(entry["last_name"] or "")
        full = first + last

        # Every supplied name part must appear in first or last name
        if not all(
            p in first or p in last or p in full
            for p in norm_parts
        ):
            continue

        # DOB filter (exact match on ISO date)
        if dob and entry["birth_date"] != dob:
            continue

        matches.append(entry)

    return matches


# --- Loading ---

def load_patient_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"\n  Error loading {path.name}: {e}\n")
        return None

def _entry_label(entry: dict) -> str:
    first = entry["first_name"] or ""
    last = entry["last_name"] or ""
    return f"{first} {last}".strip() or entry["patient_id"]


# --- Startup helpers ---

def _show_patients(registry: list[dict]) -> None:
    if not registry:
        print("  No patients found.\n")
        return
    print("  Available patients:\n")
    for entry in registry:
        name = _entry_label(entry)
        dob = entry["birth_date"] or "DOB unknown"
        gender = entry["gender"].capitalize()
        print(f"    {name:<24}  DOB: {dob}  |  {gender}")
    print()

def _check_api_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("  Warning: ANTHROPIC_API_KEY is not set.")
        print("  Add it to your .env file to enable clearance assessments.\n")
        return False
    return True


# --- Natural language command detection ---

_QUIT_WORDS = {"quit", "exit", "q", "bye", "goodbye", "done", "stop"}
_CLEAR_WORDS = {"clear", "switch", "new patient", "change patient", "start over",
                "different patient", "another patient", "load another"}
_REFRESH_WORDS = {"refresh", "redo", "reassess", "re-run", "rerun", "run again",
                  "new assessment", "redo assessment"}
_PATIENTS_WORDS = {"patients", "list", "list patients", "show patients", "who", "available"}
_HELP_WORDS = {"help", "?"}

# Explicit patient-switch prefixes the user must type when a patient is already loaded.
_SWITCH_PREFIXES = ("load ", "switch to ", "switch patient to ", "open ", "pull up ", "look up ")


# --- Pipeline ---

async def run_pipeline_on_fhir(fhir_data: dict, query: str):
    """Run the full clearance pipeline. Returns (markdown_str, ClearanceOutput, PatientSnapshot)."""
    from clearpath.fhir.normalizer import build_snapshot
    from clearpath.engines.medications import classify_medications
    from clearpath.engines.triggers import evaluate_tier1_triggers, evaluate_tier2_factors, compute_rcri
    from clearpath.engines.decision import build_clearance_output
    from clearpath.models.clinical import ScoreResult
    from clearpath.reasoning.engine import enrich_with_reasoning

    snapshot = build_snapshot(fhir_data)
    meds = classify_medications([m.name for m in snapshot.active_medications])
    snapshot.active_medications = meds

    tier1 = evaluate_tier1_triggers(snapshot, meds)
    tier2, score = evaluate_tier2_factors(snapshot, meds)
    rcri = compute_rcri(snapshot, meds)

    sr = ScoreResult(total_score=score, rcri_score=rcri, tier1_triggers=tier1, tier2_factors=tier2)
    output = build_clearance_output(snapshot, sr)
    output = await enrich_with_reasoning(output, snapshot, tier2, query)
    return output.to_markdown(), output, snapshot


# --- Main loop ---

async def main():
    print(BANNER)
    registry = build_registry()
    _check_api_key()
    _show_patients(registry)

    current_patient = None
    current_label = None
    current_output = None      # ClearanceOutput — cached after first pipeline run
    current_snapshot = None    # PatientSnapshot — cached after first pipeline run
    conversation_history = []  # list of {"q": str, "a": str}

    def _reset_conversation():
        nonlocal current_output, current_snapshot, conversation_history
        current_output = None
        current_snapshot = None
        conversation_history = []

    while True:
        try:
            prompt = f"  [{current_label}] > " if current_patient else "  > "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not line:
            continue

        low = line.lower().strip()

        # --- Meta commands ---

        if low in _QUIT_WORDS:
            print("Goodbye.")
            break

        if low in _HELP_WORDS:
            print(BANNER)
            continue

        if low in _PATIENTS_WORDS:
            _show_patients(registry)
            continue

        if low in _CLEAR_WORDS:
            current_patient = None
            current_label = None
            _reset_conversation()
            print("\n  Patient cleared. Type a name or ask about someone new.\n")
            continue

        if low in _REFRESH_WORDS:
            if current_patient is None:
                print("\n  No patient loaded. Ask about a patient to get started.\n")
                continue
            _reset_conversation()
            print(f"\n  Re-running assessment for {current_label}...")
            print("  " + "-" * 54)
            md, current_output, current_snapshot = await run_pipeline_on_fhir(current_patient, "Full clearance assessment")
            conversation_history = [{"q": "refresh", "a": md}]
            for out_line in md.splitlines():
                print("  " + out_line)
            print()
            print("  You can now ask follow-up questions about this patient.\n")
            continue

        # Legacy letter shortcuts: "patient a", "a", "load a"
        matched_legacy = False
        for key, filename in LEGACY_KEYS.items():
            if low in (f"patient {key}", key, f"load {key}"):
                path = PATIENTS_DIR / filename
                data = load_patient_file(path)
                if data:
                    current_patient = data
                    _reset_conversation()
                    p = data.get("patient", {})
                    name_list = p.get("name", [])
                    ne = name_list[0] if name_list else {}
                    given = (ne.get("given") or [""])[0]
                    last = ne.get("family", "")
                    current_label = f"{given} {last}".strip() or key.upper()
                    print(f"\n  Patient loaded: {current_label}. Ask a question or request a clearance check.\n")
                matched_legacy = True
                break

        if matched_legacy:
            continue

        # Explicit patient switch when a patient is already loaded.
        if current_patient is not None:
            switch_match = next((p for p in _SWITCH_PREFIXES if low.startswith(p)), None)
            if switch_match:
                line = line[len(switch_match):].strip()
                low = line.lower().strip()
                current_patient = None
                current_label = None
                _reset_conversation()
                # fall through to name-lookup branch below

        # Name-only lookup. Only runs when no patient is loaded — otherwise treat input as follow-up.
        if current_patient is None and _looks_like_name_query(line):
            name_str, dob = _extract_dob(line)
            name_str = name_str.strip()
            if name_str:
                matches = find_patients(registry, name_str, dob)

                if not matches:
                    dob_hint = f" (DOB {dob})" if dob else ""
                    print(f"\n  No patient found matching '{name_str}'{dob_hint}.\n"
                          f"  Type 'patients' to see who is available.\n")
                    continue

                if len(matches) == 1:
                    data = load_patient_file(matches[0]["file"])
                    if data:
                        current_patient = data
                        _reset_conversation()
                        current_label = _entry_label(matches[0])
                        print(f"\n  Patient loaded: {current_label}  (DOB: {matches[0]['birth_date']})")
                        print(f"  Ask a question or say \"run clearance\" to start the assessment.\n")
                    continue

                # Multiple matches — disambiguate
                print(f"\n  Multiple patients found — please select:\n")
                for i, entry in enumerate(matches, 1):
                    name = _entry_label(entry)
                    print(f"  [{i}]  {name:<20}  DOB: {entry['birth_date']}  {entry['gender'].capitalize()}")
                print()
                try:
                    sel = input("  Enter number: ").strip()
                    idx = int(sel) - 1
                    if 0 <= idx < len(matches):
                        chosen = matches[idx]
                        data = load_patient_file(chosen["file"])
                        if data:
                            current_patient = data
                            _reset_conversation()
                            current_label = _entry_label(chosen)
                            print(f"\n  Patient loaded: {current_label}  (DOB: {chosen['birth_date']})")
                            print(f"  Ask a question or say \"run clearance\" to start the assessment.\n")
                    else:
                        print("\n  Invalid selection.\n")
                except (ValueError, EOFError, KeyboardInterrupt):
                    print("\n  Cancelled.\n")
                continue

        # Clinical query — if no patient loaded, try to find a name in the question
        if current_patient is None:
            inline_matches = find_patient_in_query(registry, line)

            if not inline_matches:
                print("\n  I don't have a patient loaded yet. Ask about someone by name,")
                print("  or type 'patients' to see who is available.\n")
                continue

            # Disambiguate if multiple
            if len(inline_matches) > 1:
                print(f"\n  Multiple patients found in your question — please select:\n")
                for i, entry in enumerate(inline_matches, 1):
                    name = _entry_label(entry)
                    print(f"  [{i}]  {name:<20}  DOB: {entry['birth_date']}  {entry['gender'].capitalize()}")
                print()
                try:
                    sel = input("  Enter number: ").strip()
                    idx = int(sel) - 1
                    if 0 <= idx < len(inline_matches):
                        chosen = inline_matches[idx]
                    else:
                        print("\n  Invalid selection.\n")
                        continue
                except (ValueError, EOFError, KeyboardInterrupt):
                    print("\n  Cancelled.\n")
                    continue
            else:
                chosen = inline_matches[0]

            label = _entry_label(chosen)
            dob = chosen["birth_date"]
            gender = chosen["gender"].capitalize()
            print(f"\n  Found patient: {label}  |  DOB: {dob}  |  {gender}")
            try:
                confirm = input("  Proceed with this patient? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                continue

            if confirm in ("n", "no"):
                print("\n  Cancelled. Type a patient name to load one manually.\n")
                continue

            data = load_patient_file(chosen["file"])
            if not data:
                continue
            current_patient = data
            _reset_conversation()
            current_label = label
            # Fall through — run the pipeline with the original question

        # Clinical query dispatch
        if current_output is None:
            # First query — run full pipeline
            print(f"\n  Running clearance assessment for {current_label}...")
            print("  " + "-" * 54)
            md, current_output, current_snapshot = await run_pipeline_on_fhir(current_patient, line)
            conversation_history = [{"q": line, "a": md}]
            for out_line in md.splitlines():
                print("  " + out_line)
            print()
            print("  You can now ask follow-up questions about this patient.\n")
        else:
            # Follow-up — streamed, with prompt caching on system + chart context
            from clearpath.reasoning.engine import stream_followup
            print(f"\n  ClearPath >")
            print("  ", end="", flush=True)
            accumulated = ""
            async for chunk in stream_followup(current_output, current_snapshot, line, conversation_history):
                accumulated += chunk
                # Maintain the 2-space left indent across streamed newlines.
                print(chunk.replace("\n", "\n  "), end="", flush=True)
            print("\n")
            conversation_history.append({"q": line, "a": accumulated})


asyncio.run(main())
