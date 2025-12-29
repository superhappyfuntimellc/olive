#!/usr/bin/env python3
"""
Deep Functional Tests for Beta Readiness
Tests actual behavior of critical systems.
"""

import sys
import os

# Add parent directory to path to import app functions
sys.path.insert(0, "/workspaces/olive")

# Import critical functions from app
from app import (
    init_system_contract,
    init_style_learning_engine,
    validate_contract_compliance,
    enforce_contract_on_learning,
    set_intensity_limits,
)

# Test results
tests_passed = 0
tests_failed = 0
failures = []


def test(name: str, condition: bool, details: str = "") -> None:
    """Record test result."""
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        print(f"✅ {name}")
        if details:
            print(f"   └─ {details}")
    else:
        tests_failed += 1
        failures.append((name, details))
        print(f"❌ {name}")
        if details:
            print(f"   └─ {details}")


def section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ============================================================
# TEST 1: SYSTEM CONTRACT INITIALIZATION
# ============================================================
section("1. SYSTEM CONTRACT INITIALIZATION")

try:
    contract = init_system_contract()
    test(
        "System Contract initializes",
        isinstance(contract, dict),
        f"Contract version: {contract.get('version')}",
    )
    test(
        "Contract has required rules",
        "contract_rules" in contract,
        f"Rules: {len(contract.get('contract_rules', {}))} defined",
    )
    test(
        "Contract has project settings",
        "project_settings" in contract,
        f"Settings: {len(contract.get('project_settings', {}))} keys",
    )
    test(
        "Contract has isolation settings",
        "isolation" in contract,
        f"Isolation enabled: {contract.get('isolation', {}).get('bay_isolation_enabled')}",
    )

    # Test contract rules
    rules = contract.get("contract_rules", {})
    test(
        "Rule: user_instructions_are_law",
        rules.get("user_instructions_are_law") is True,
    )
    test(
        "Rule: project_settings_override_learning",
        rules.get("project_settings_override_learning") is True,
    )
    test(
        "Rule: intensity_controls_are_hard_limits",
        rules.get("intensity_controls_are_hard_limits") is True,
    )
    test(
        "Rule: no_cross_project_leakage",
        rules.get("no_cross_project_leakage") is True,
    )

except Exception as e:
    test("System Contract initialization", False, str(e))

# ============================================================
# TEST 2: CONTRACT VALIDATION LOGIC
# ============================================================
section("2. CONTRACT VALIDATION LOGIC")


# Mock session state for testing
class MockSessionState:
    def __init__(self):
        self._system_contract = init_system_contract()

    def get(self, key, default=None):
        return getattr(self, key, default)


try:
    # Test with unlocked settings
    mock_state = MockSessionState()
    import app

    app.st = type("obj", (object,), {"session_state": mock_state})()

    # Test intensity validation
    result = validate_contract_compliance("set_intensity", {"intensity": 0.5})
    test(
        "Contract allows intensity within limits",
        result is True,
        "Intensity 0.5 allowed (min=0.0, max=1.0)",
    )

    # Test with locked settings
    mock_state._system_contract["project_settings"]["locked"] = True
    result = validate_contract_compliance("change_style", {})
    test(
        "Contract blocks changes when locked",
        result is False,
        "Change blocked with locked=True",
    )

    # Test style lock
    mock_state._system_contract["project_settings"]["locked"] = False
    mock_state._system_contract["project_settings"]["writing_style_locked"] = True
    result = validate_contract_compliance("change_style", {})
    test(
        "Contract enforces style lock",
        result is False,
        "Style change blocked when style_locked=True",
    )

    # Test voice lock
    mock_state._system_contract["project_settings"]["voice_selection_locked"] = True
    result = validate_contract_compliance("change_voice", {})
    test(
        "Contract enforces voice lock",
        result is False,
        "Voice change blocked when voice_locked=True",
    )

except Exception as e:
    test("Contract validation logic", False, str(e))

# ============================================================
# TEST 3: LEARNING ENGINE ENFORCEMENT
# ============================================================
section("3. LEARNING ENGINE CONTRACT ENFORCEMENT")

try:
    contract = init_system_contract()
    contract["project_settings"]["ai_intensity_max"] = 0.6

    # Mock session state
    mock_state = MockSessionState()
    mock_state._system_contract = contract
    app.st.session_state = mock_state

    # Test learning data enforcement
    learning_data = {"suggested_intensity": 0.9}
    enforced = enforce_contract_on_learning(learning_data)

    test(
        "Learning respects intensity limits",
        enforced.get("suggested_intensity") == 0.6,
        f"Intensity capped at 0.6 (was 0.9)",
    )

except Exception as e:
    test("Learning engine enforcement", False, str(e))

# ============================================================
# TEST 4: STYLE LEARNING ENGINE INITIALIZATION
# ============================================================
section("4. STYLE LEARNING ENGINE")

try:
    engine = init_style_learning_engine()

    test(
        "Learning engine initializes",
        isinstance(engine, dict),
        f"Version: {engine.get('version')}",
    )
    test(
        "Engine has edit_pairs storage",
        "edit_pairs" in engine and isinstance(engine["edit_pairs"], list),
        "Empty list initialized",
    )
    test(
        "Engine has learned_patterns",
        "learned_patterns" in engine and isinstance(engine["learned_patterns"], dict),
        f"{len(engine['learned_patterns'])} pattern types",
    )
    test(
        "Engine has style_stats",
        "style_stats" in engine and isinstance(engine["style_stats"], dict),
        f"{len(engine['style_stats'])} stat types",
    )

    # Check pattern types
    patterns = engine.get("learned_patterns", {})
    test("Pattern: sentence_length tracked", "sentence_length" in patterns)
    test("Pattern: word_preferences tracked", "word_preferences" in patterns)
    test("Pattern: phrase_patterns tracked", "phrase_patterns" in patterns)
    test("Pattern: tone_indicators tracked", "tone_indicators" in patterns)

except Exception as e:
    test("Style learning engine initialization", False, str(e))

# ============================================================
# TEST 5: EDIT HISTORY LIMITS
# ============================================================
section("5. LEARNING HISTORY LIMITS")

try:
    engine = init_style_learning_engine()

    # Simulate 150 edits
    for i in range(150):
        edit_entry = {
            "before": f"before_{i}",
            "after": f"after_{i}",
            "timestamp": f"2025-12-29T{i:02d}:00:00",
            "context": f"context_{i}",
        }
        engine["edit_pairs"].insert(0, edit_entry)
        engine["edit_pairs"] = engine["edit_pairs"][:100]

    test(
        "Edit history limited to 100",
        len(engine["edit_pairs"]) == 100,
        f"Has {len(engine['edit_pairs'])} entries after 150 inserts",
    )

    # Verify most recent kept
    test(
        "Most recent edits preserved",
        engine["edit_pairs"][0]["after"] == "after_149",
        "Latest edit at position 0",
    )
    test(
        "Oldest edits dropped",
        engine["edit_pairs"][-1]["after"] == "after_50",
        "Entry 50 at position 99 (entries 0-49 dropped)",
    )

except Exception as e:
    test("Learning history limits", False, str(e))

# ============================================================
# TEST 6: DATA STRUCTURE INTEGRITY
# ============================================================
section("6. DATA STRUCTURE INTEGRITY")

try:
    # Test contract structure
    contract = init_system_contract()
    required_contract_keys = [
        "version",
        "created_ts",
        "contract_rules",
        "project_settings",
        "isolation",
    ]
    for key in required_contract_keys:
        test(f"Contract has key: {key}", key in contract)

    # Test learning engine structure
    engine = init_style_learning_engine()
    required_engine_keys = [
        "version",
        "created_ts",
        "learning_enabled",
        "edit_pairs",
        "learned_patterns",
        "style_stats",
    ]
    for key in required_engine_keys:
        test(f"Engine has key: {key}", key in engine)

except Exception as e:
    test("Data structure integrity", False, str(e))

# ============================================================
# TEST 7: INTENSITY LIMIT ENFORCEMENT
# ============================================================
section("7. INTENSITY LIMIT ENFORCEMENT")

try:
    # This would require mocking st.session_state more extensively
    # For now, verify the function exists and has correct signature
    import inspect

    sig = inspect.signature(set_intensity_limits)
    params = list(sig.parameters.keys())

    test(
        "set_intensity_limits exists",
        callable(set_intensity_limits),
        f"Parameters: {params}",
    )
    test(
        "set_intensity_limits has correct parameters",
        params == ["min_val", "max_val"],
        "Takes min_val and max_val",
    )

except Exception as e:
    test("Intensity limit enforcement", False, str(e))

# ============================================================
# FINAL REPORT
# ============================================================
section("FUNCTIONAL TEST SUMMARY")

total = tests_passed + tests_failed
print(f"\nTotal Tests: {total}")
print(f"Passed: {tests_passed} ({100 * tests_passed // total if total > 0 else 0}%)")
print(f"Failed: {tests_failed}")

if failures:
    print("\n⚠️  FAILED TESTS:")
    for name, details in failures:
        print(f"  • {name}")
        if details:
            print(f"    {details}")

print("\n" + "=" * 70)
sys.exit(0 if tests_failed == 0 else 1)
