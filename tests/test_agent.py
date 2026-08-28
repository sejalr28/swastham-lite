"""
test_agent.py
--------------
Regression tests for the agent's tool-routing logic.

Includes the exact collision case found during manual testing in week 2:
a message mentioning "bedtime" (generic) alongside a caffeine/wind-down
specific phrase was incorrectly routed to bedtime_calculator because of
keyword-count ties. Fixed by scoring on matched-phrase length instead of
hit count, plus removing overly generic keywords. These cases must never
silently regress.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Agent  # noqa: E402


# (message, expected_mode, expected_tool_name_or_None)
ROUTING_CASES = [
    ("What time should I go to bed if I need to wake up at 6:30am?", "tool", "bedtime_calculator"),
    ("When should I stop drinking coffee if my bedtime is 10:30pm?", "tool", "caffeine_cutoff_calculator"),
    ("Can you build me a wind-down routine? My bedtime is 11pm", "tool", "build_winddown_routine"),
    ("What's the last time I should have caffeine if I sleep at 9pm?", "tool", "caffeine_cutoff_calculator"),
    ("What time should I go to bed?", "clarify", "bedtime_calculator"),  # missing wake time
    ("How does caffeine affect sleep quality?", "rag", None),
    ("What are the stages of sleep?", "rag", None),
    ("I've been having thoughts of ending my life", "crisis", None),
    # Ambiguous: two distinct times present -> must ask for clarification,
    # not silently pick the wrong one (this is the bug found in dev testing:
    # the coffee time 10:30pm was wrongly used as the bedtime).
    ("Is 10:30pm too late for my last coffee if bedtime is midnight?", "clarify", "caffeine_cutoff_calculator"),
    ("My wake time is 6am, what sleep cycles work best?", "tool", "bedtime_calculator"),
]


def test_routing_cases():
    agent = Agent()
    failures = []
    for message, expected_mode, expected_tool in ROUTING_CASES:
        resp = agent.handle(message)
        if resp.mode != expected_mode or (expected_tool and resp.tool_name != expected_tool):
            failures.append({
                "message": message,
                "expected": (expected_mode, expected_tool),
                "got": (resp.mode, resp.tool_name),
            })
    assert not failures, f"Routing mismatches: {failures}"


def test_tool_outputs_have_no_errors_on_valid_input():
    agent = Agent()
    valid_messages = [
        "What time should I go to bed if I need to wake up at 6:30am?",
        "When should I stop drinking coffee if my bedtime is 10:30pm?",
        "Can you build me a wind-down routine? My bedtime is 11pm",
    ]
    failures = []
    for message in valid_messages:
        resp = agent.handle(message)
        if resp.tool_result and "error" in resp.tool_result:
            failures.append((message, resp.tool_result))
    assert not failures, f"Unexpected tool errors on valid input: {failures}"


def _run_manually():
    passed, failed = 0, 0
    for fn in [test_routing_cases, test_tool_outputs_have_no_errors_on_valid_input]:
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
