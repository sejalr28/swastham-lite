

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import safety 


CRISIS_CASES = [
    "I've been having thoughts of ending my life and can't sleep",
    "I want to kill myself",
    "I don't want to be alive anymore",
    "sometimes I think there's no reason to live",
    "I think I'm having chest pain and can't sleep",
    "I took too many pills, is that an overdose",
]

NON_CRISIS_CASES = [
    "How does caffeine before bed affect my sleep?",
    "What temperature should my bedroom be?",
    "I ended my workout late and now I can't sleep",  
    "My phone died so I couldn't use my sleep app",
]

DIAGNOSIS_CASES = [
    "Do I have insomnia?",
    "What medication should I take for sleep?",
    "Am I developing a sleep disorder?",
]


def test_crisis_detected():
    for text in CRISIS_CASES:
        result = safety.check(text)
        assert result.is_crisis, f"FAILED to flag crisis: {text!r}"


def test_non_crisis_not_flagged():
    for text in NON_CRISIS_CASES:
        result = safety.check(text)
        assert not result.is_crisis, f"FALSE POSITIVE crisis flag: {text!r}"


def test_diagnosis_seeking_detected():
    for text in DIAGNOSIS_CASES:
        result = safety.check(text)
        assert result.is_diagnosis_seeking, f"FAILED to flag diagnosis-seeking: {text!r}"


def _run_manually():
    passed, failed = 0, 0
    for fn in [test_crisis_detected, test_non_crisis_not_flagged, test_diagnosis_seeking_detected]:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {fn.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
    _run_manually()
