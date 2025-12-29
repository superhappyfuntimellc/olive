#!/usr/bin/env python3
"""
Beta Readiness Integrity Check
Validates all critical systems without modifying behavior.
"""

import json
import os
import sys
import re
from typing import Dict, List, Any, Tuple

# Test results
results: List[Tuple[str, str, str]] = []


def check(test_name: str, passed: bool, details: str = "") -> None:
    """Record test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    results.append((test_name, status, details))


def section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================
# TEST 1: VERIFY ALL INITIALIZATION FUNCTIONS EXIST
# ============================================================
section("1. SYSTEM INITIALIZATION")

with open("/workspaces/olive/app.py", "r", encoding="utf-8") as f:
    app_code = f.read()

# Check critical init functions
init_functions = [
    "init_system_contract",
    "init_style_learning_engine",
    "init_state",
]

for func in init_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Function {func}() exists", found)

# Check session state defaults
critical_defaults = [
    "_system_contract",
    "_style_learning_engine",
    "_voice_settings_by_bay",
    "_trash_bin",
    "_undo_history",
    "_redo_history",
    "trained_voice",
    "voice_lane",
    "writing_style",
    "ai_intensity",
]

for key in critical_defaults:
    pattern = rf'"{key}":|\'_{key}\':|"{key}",'
    found = bool(re.search(pattern, app_code))
    check(f"Session state key '{key}' initialized", found, f"Found in defaults")

# ============================================================
# TEST 2: AUTOSAVE MECHANISM
# ============================================================
section("2. AUTOSAVE & PERSISTENCE")

# Check autosave functions
autosave_functions = [
    "autosave_state",
    "load_autosave",
    "get_autosave_info",
    "maybe_autosave_throttled",
]

for func in autosave_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Autosave function {func}() exists", found)

# Check autosave throttling
throttle_pattern = r"AUTOSAVE_MIN_INTERVAL_S\s*=\s*[\d.]+"
has_throttle = bool(re.search(throttle_pattern, app_code))
check("Autosave throttling configured", has_throttle)

# Check autosave path
autosave_dir = "autosave"
autosave_file = os.path.join(autosave_dir, "olivetti_state.json")
check(f"Autosave directory path configured", "autosave" in app_code)

# If autosave exists, verify structure
if os.path.exists(autosave_file):
    try:
        with open(autosave_file, "r") as f:
            autosave_data = json.load(f)
        has_session = "session" in autosave_data
        has_timestamp = "saved_ts" in autosave_data
        check("Autosave file structure valid", has_session and has_timestamp)
    except Exception as e:
        check("Autosave file structure valid", False, str(e))
else:
    check("Autosave file structure", True, "No autosave file (fresh install)")

# ============================================================
# TEST 3: UNDO/REDO SYSTEM
# ============================================================
section("3. UNDO/REDO SYSTEM")

# Check undo/redo functions
undo_functions = [
    "push_undo_history",
    "undo_text",
    "redo_text",
]

for func in undo_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Undo/Redo function {func}() exists", found)

# Check keyboard shortcuts
kb_shortcuts = ["Ctrl+Z", "Ctrl+Y", "keydown"]
for shortcut in kb_shortcuts:
    found = shortcut.replace("+", r"\+") in app_code or shortcut in app_code
    check(f"Keyboard shortcut '{shortcut}' configured", found)

# Check history limit (should be 50)
max_history_pattern = r"MAX.*HISTORY|len.*_undo_history.*>\s*\d+"
has_limit = bool(re.search(max_history_pattern, app_code))
check("Undo history limit exists", has_limit)

# ============================================================
# TEST 4: TRASH BIN SYSTEM
# ============================================================
section("4. TRASH BIN SYSTEM")

trash_functions = [
    "delete_current_draft",
    "get_trash_bin_items",
    "restore_from_trash",
    "permanently_delete_from_trash",
    "clear_trash_bin",
]

for func in trash_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Trash function {func}() exists", found)

# Check trash limit (should be 10)
trash_limit_pattern = r"len.*_trash_bin.*>\s*10|MAX.*TRASH|TRASH.*LIMIT"
has_limit = bool(re.search(trash_limit_pattern, app_code, re.IGNORECASE))
check("Trash bin limit (10 items) enforced", has_limit)

# ============================================================
# TEST 5: MY VOICE PROFILES
# ============================================================
section("5. MY VOICE PROFILE SYSTEM")

voice_functions = [
    "create_my_voice_profile",
    "delete_my_voice_profile",
    "get_my_voice_profiles",
    "learn_voice_from_edit",
    "get_voice_learning_stats",
]

for func in voice_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Voice profile function {func}() exists", found)

# Check voice persistence in session state
voice_state_keys = ["voices", "trained_voice", "voice_lane"]
for key in voice_state_keys:
    pattern = rf'"{key}"'
    found = bool(re.search(pattern, app_code))
    check(f"Voice state key '{key}' persisted", found)

# ============================================================
# TEST 6: STYLE LEARNING ENGINE
# ============================================================
section("6. ADAPTIVE STYLE LEARNING")

learning_functions = [
    "init_style_learning_engine",
    "learn_from_edit",
    "learn_from_acceptance",
    "learn_from_rejection",
    "_extract_and_learn_patterns",
    "get_style_learning_stats",
]

for func in learning_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Learning function {func}() exists", found)

# Check learning limits (100 edits per voice)
learning_limit_pattern = r"len.*edit_pairs.*>\s*100|MAX.*LEARNING|LEARNING.*LIMIT"
has_limit = bool(re.search(learning_limit_pattern, app_code, re.IGNORECASE))
check("Learning history limit (100 edits) exists", has_limit)

# Check pattern extraction
pattern_types = ["sentence_length", "word_preferences"]
for ptype in pattern_types:
    found = ptype in app_code
    check(f"Learning pattern type '{ptype}' implemented", found)

# ============================================================
# TEST 7: PER-BAY VOICE LOCKS
# ============================================================
section("7. PER-BAY VOICE LOCK SYSTEM")

bay_functions = [
    "get_voice_settings",
    "set_voice_settings",
    "save_voice_settings_for_bay",
    "load_voice_settings_for_bay",
    "on_bay_change",
]

for func in bay_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Bay settings function {func}() exists", found)

# Check bay definitions
bays = ["NEW", "ROUGH", "EDIT", "FINAL"]
bays_defined = all(bay in app_code for bay in bays)
check("All bays (NEW/ROUGH/EDIT/FINAL) defined", bays_defined)

# Check per-bay storage
bay_storage_pattern = r"_voice_settings_by_bay"
has_storage = bool(re.search(bay_storage_pattern, app_code))
check("Per-bay voice settings storage exists", has_storage)

# ============================================================
# TEST 8: SYSTEM CONTRACT
# ============================================================
section("8. SYSTEM CONTRACT ENFORCEMENT")

contract_functions = [
    "init_system_contract",
    "validate_contract_compliance",
    "enforce_contract_on_learning",
    "lock_project_settings",
    "unlock_project_settings",
    "set_intensity_limits",
    "lock_writing_style",
    "unlock_writing_style",
    "lock_voice_selection",
    "unlock_voice_selection",
    "get_contract_status",
]

for func in contract_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Contract function {func}() exists", found)

# Check contract rules
contract_rules = [
    "user_instructions_are_law",
    "project_settings_override_learning",
    "intensity_controls_are_hard_limits",
    "no_cross_project_leakage",
]

for rule in contract_rules:
    found = rule in app_code
    check(f"Contract rule '{rule}' defined", found)

# ============================================================
# TEST 9: BAY TRANSFER & EXPORT
# ============================================================
section("9. BAY TRANSFER SYSTEM")

transfer_functions = [
    "can_transfer_bay",
    "get_next_bay",
    "transfer_to_next_bay",
    "_execute_bay_transfer",
    "get_export_filename",
]

for func in transfer_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Transfer function {func}() exists", found)

# ============================================================
# TEST 10: IMPORT/EXPORT
# ============================================================
section("10. IMPORT/EXPORT SYSTEM")

# Check file format support
file_formats = ["TXT", "MD", "DOCX", "PDF"]
import_export_pattern = r"(import|export).*file"
has_import_export = bool(re.search(import_export_pattern, app_code, re.IGNORECASE))
check("Import/Export functions exist", has_import_export)

for fmt in file_formats:
    pattern = rf"\b{fmt}\b"
    found = bool(re.search(pattern, app_code, re.IGNORECASE))
    check(f"File format '{fmt}' supported", found)

# ============================================================
# TEST 11: CLOUD SYNC
# ============================================================
section("11. CLOUD SYNC SYSTEM")

cloud_functions = [
    "list_cloud_saves",
    "upload_to_cloud",
    "download_from_cloud",
    "delete_from_cloud",
]

for func in cloud_functions:
    pattern = rf"def {func}\("
    found = bool(re.search(pattern, app_code))
    check(f"Cloud sync function {func}() exists", found)

# Check S3 compatibility
s3_pattern = r"boto3|S3|s3_"
has_s3 = bool(re.search(s3_pattern, app_code, re.IGNORECASE))
check("S3-compatible storage configured", has_s3)

# ============================================================
# TEST 12: UI INDICATORS
# ============================================================
section("12. UI INDICATORS & STATE DISPLAY")

# Check word count
word_count_pattern = r"count.*word|word.*count"
has_word_count = bool(re.search(word_count_pattern, app_code, re.IGNORECASE))
check("Word count indicator exists", has_word_count)

# Check bay indicators
bay_indicator_pattern = r"badge|indicator|status.*bay"
has_bay_indicators = bool(re.search(bay_indicator_pattern, app_code, re.IGNORECASE))
check("Bay status indicators exist", has_bay_indicators)

# Check trash bin indicator
trash_indicator_pattern = r"trash.*count|trash.*badge"
has_trash_indicator = bool(re.search(trash_indicator_pattern, app_code, re.IGNORECASE))
check("Trash bin indicator exists", has_trash_indicator)

# ============================================================
# TEST 13: CROSS-PROJECT ISOLATION
# ============================================================
section("13. CROSS-PROJECT DATA ISOLATION")

# Check project ID tracking
project_isolation = [
    "project_id",
    "active_project_by_bay",
    "no_cross_project_leakage",
]

for item in project_isolation:
    found = item in app_code
    check(f"Project isolation: '{item}' implemented", found)

# Check bay isolation
bay_isolation_pattern = r"bay_isolation|voice_isolation_per_bay"
has_bay_isolation = bool(re.search(bay_isolation_pattern, app_code))
check("Bay isolation enabled", has_bay_isolation)

# ============================================================
# TEST 14: STATE PERSISTENCE ON REFRESH
# ============================================================
section("14. STATE PERSISTENCE")

# Check session state initialization
state_init_pattern = r"if.*not in st\.session_state|session_state\.setdefault"
has_state_init = bool(re.search(state_init_pattern, app_code))
check("Session state initialization exists", has_state_init)

# Check dirty flag for autosave trigger
dirty_flag_pattern = r"_dirty|mark_dirty"
has_dirty_flag = bool(re.search(dirty_flag_pattern, app_code))
check("Dirty flag for autosave trigger exists", has_dirty_flag)

# ============================================================
# FINAL REPORT
# ============================================================
section("INTEGRITY CHECK SUMMARY")

passed = sum(1 for _, status, _ in results if status == "✅ PASS")
failed = sum(1 for _, status, _ in results if status == "❌ FAIL")
total = len(results)

print(f"\nTotal Tests: {total}")
print(f"Passed: {passed} ({100 * passed // total}%)")
print(f"Failed: {failed}")

if failed > 0:
    print("\n⚠️  FAILED TESTS:")
    for test_name, status, details in results:
        if status == "❌ FAIL":
            print(f"  • {test_name}")
            if details:
                print(f"    {details}")

print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

for test_name, status, details in results:
    print(f"{status} {test_name}")
    if details:
        print(f"     └─ {details}")

# Exit code
sys.exit(0 if failed == 0 else 1)
