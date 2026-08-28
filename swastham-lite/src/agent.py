"""
agent.py
--------
The agent layer that sits in front of rag.py. Its job: for each incoming
message, decide whether to:
  1. call a deterministic tool (bedtime calc, caffeine cutoff, wind-down routine)
  2. fall back to RAG retrieval + generation for general questions
  3. ask a clarifying question when a tool clearly applies but required
     parameters (e.g. a time) are missing from the message

Two-stage routing, both intentionally simple and inspectable for week 2:
  - Stage 1: keyword-based intent match against TOOL_REGISTRY (see tools.py)
  - Stage 2: regex-based slot extraction (times in HH:MM or "10pm" style)

This is NOT an LLM-driven planner yet -- it's a transparent, testable
router. The interface (`Agent.handle(message) -> AgentResponse`) is
designed so a future version could swap this logic for real LLM
function-calling (e.g. Claude's tool-use API) without changing callers.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from tools import TOOL_REGISTRY
from rag import RAGPipeline, RAGResponse
import safety


TIME_PATTERN = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b|\b(\d{1,2})\s*(am|pm)\b",
    re.IGNORECASE,
)

NAMED_TIMES = {"midnight": "00:00", "noon": "12:00", "midday": "12:00"}
NAMED_TIME_PATTERN = re.compile(r"\b(midnight|noon|midday)\b", re.IGNORECASE)


def _compile_patterns() -> None:
    """Compiles each tool's raw regex strings once at import time."""
    for spec in TOOL_REGISTRY.values():
        spec["patterns"] = [re.compile(p, re.IGNORECASE) for p in spec["pattern_strings"]]


def _extract_all_times(text: str) -> List[str]:
    """
    Finds every distinct time-like expression in text (numeric "10:30pm"
    style plus named times like "midnight"/"noon"), normalized to HH:MM
    24h, in order of appearance.

    Returning *all* matches (not just the first) matters: a message like
    "Is 10:30pm too late for coffee if my bedtime is midnight?" contains
    two distinct times. Picking the first one silently gives a WRONG
    answer (it would treat the coffee time as the bedtime). See
    tests/test_agent.py for the case that caught this during dev testing.
    """
    found: List[tuple] = []

    for m in TIME_PATTERN.finditer(text):
        if m.group(1):
            hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        else:
            hour, minute, ampm = int(m.group(4)), 0, m.group(5)
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            found.append((m.start(), f"{hour:02d}:{minute:02d}"))

    for m in NAMED_TIME_PATTERN.finditer(text):
        found.append((m.start(), NAMED_TIMES[m.group(1).lower()]))

    found.sort(key=lambda x: x[0])
    return [t for _, t in found]




def _match_tool(message: str) -> Optional[str]:
    """
    Stage 1: regex-pattern match against each tool's pattern list, scored
    by total matched-text length. Regex (rather than plain substrings)
    lets us catch paraphrases like "last time I should have caffeine"
    without resorting to generic single-word keywords like "caffeine"
    that would misfire on plain informational questions (e.g. "how does
    caffeine affect sleep quality?", which should go to RAG, not a tool).
    """
    best_tool, best_score = None, 0
    for tool_name, spec in TOOL_REGISTRY.items():
        score = 0
        for pattern in spec["patterns"]:
            m = pattern.search(message)
            if m:
                score += len(m.group(0))
        if score > best_score:
            best_tool, best_score = tool_name, score
    return best_tool


@dataclass
class AgentResponse:
    answer: str
    mode: str  # "tool" | "rag" | "clarify" | "crisis"
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    sources: List[Dict] = field(default_factory=list)


class Agent:
    def __init__(self):
        _compile_patterns()
        self.rag = RAGPipeline()

    def handle(self, message: str) -> AgentResponse:
        # Crisis check first, always -- bypasses tool routing and RAG entirely.
        safety_result = safety.check(message)
        if safety_result.is_crisis:
            return AgentResponse(answer=safety_result.fixed_response, mode="crisis")

        tool_name = _match_tool(message)

        if tool_name in ("bedtime_calculator", "caffeine_cutoff_calculator", "build_winddown_routine"):
            times = _extract_all_times(message)
            slot_label = "wake time" if tool_name == "bedtime_calculator" else "bedtime"

            if not times:
                prompts = {
                    "bedtime_calculator": "Happy to help you figure out a bedtime - what time do you need to wake up? (e.g. '7:00am')",
                    "caffeine_cutoff_calculator": "I can work that out - what time are you planning to go to bed? (e.g. '10:30pm')",
                    "build_winddown_routine": "I can build you a wind-down routine - what time is your bedtime? (e.g. '11:00pm')",
                }
                return AgentResponse(answer=prompts[tool_name], mode="clarify", tool_name=tool_name)

            if len(times) > 1:
                # Ambiguous: don't guess which time is the relevant slot.
                # This is the fix for a bug found in dev testing where a
                # message with two times (e.g. "coffee at 10:30pm if bedtime
                # is midnight") silently used the wrong one.
                return AgentResponse(
                    answer=(f"I found more than one time in your message ({', '.join(times)}) "
                            f"- could you confirm which one is your {slot_label}?"),
                    mode="clarify",
                    tool_name=tool_name,
                )

            slot_value = times[0]
            kwarg = "wake_time" if tool_name == "bedtime_calculator" else "bedtime"
            result = TOOL_REGISTRY[tool_name]["fn"](**{kwarg: slot_value})
            return self._tool_response(tool_name, result)

        # No tool matched strongly enough -> fall back to RAG.
        rag_response: RAGResponse = self.rag.answer(message)
        return AgentResponse(
            answer=rag_response.answer,
            mode="crisis" if rag_response.flagged_crisis else "rag",
            sources=rag_response.sources,
        )

    @staticmethod
    def _tool_response(tool_name: str, result: Dict[str, Any]) -> AgentResponse:
        if "error" in result:
            return AgentResponse(
                answer=f"I couldn't do that: {result['error']}",
                mode="clarify",
                tool_name=tool_name,
                tool_result=result,
            )

        # Turn structured tool output into a short natural-language line.
        if tool_name == "bedtime_calculator":
            text = (f"To wake up at {result['wake_time']}, aim to be asleep by around "
                    f"{result['recommended_bedtime']} ({result['total_sleep_hours']}h across "
                    f"{result['cycles']} sleep cycles). {result['note']}")
        elif tool_name == "caffeine_cutoff_calculator":
            text = (f"For a {result['bedtime']} bedtime, try to have your last caffeine by "
                    f"around {result['suggested_caffeine_cutoff']}. {result['note']}")
        elif tool_name == "build_winddown_routine":
            steps = "; ".join(f"{s['time']} - {s['activity']}" for s in result["schedule"])
            text = f"Here's a wind-down routine starting at {result['routine_start']}: {steps}. {result['note']}"
        else:
            text = str(result)

        return AgentResponse(answer=text, mode="tool", tool_name=tool_name, tool_result=result)


if __name__ == "__main__":
    agent = Agent()
    demo_messages = [
        "What time should I go to bed if I need to wake up at 6:30am?",
        "When should I stop drinking coffee if my bedtime is 10:30pm?",
        "Can you build me a wind-down routine? My bedtime is 11pm",
        "What time should I go to bed?",  # missing wake time -> clarify
        "How does caffeine affect sleep quality?",  # -> RAG
        "I've been having thoughts of ending my life",  # -> crisis
    ]
    for msg in demo_messages:
        print("=" * 70)
        print("USER:", msg)
        resp = agent.handle(msg)
        print(f"[{resp.mode}]", resp.answer)
