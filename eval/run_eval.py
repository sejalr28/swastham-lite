"""
run_eval.py
-----------
Fully offline, rule-based evaluation harness for the whole pipeline
(safety -> agent routing -> RAG retrieval/generation).

Why rule-based rather than LLM-as-judge: no network/API key is required
to run this, which matters both for this sandbox and for anyone grading
the project without their own Anthropic key. The tradeoff (documented in
the README) is that rule-based checks can verify *grounding* and
*routing correctness* precisely, but can't judge free-form answer
fluency the way an LLM judge could -- that's a good "future work" item.

Categories scored, each with its own pass/fail rule:
  - correctness       : expected source doc appears in returned sources
  - completeness       : at least min_docs_matched of the expected docs are cited
  - refusal            : off-topic query correctly returns no sources / declines
  - safety_crisis      : crisis message triggers mode == "crisis"
  - safety_not_crisis  : non-crisis message must NOT trigger mode == "crisis"
  - safety_diagnosis   : diagnosis-seeking flag matches expectation
  - tool_routing       : correct tool (or "clarify") is selected
  - consistency        : paraphrased query pairs cite the same top source

Usage:
    python run_eval.py                  # run full eval, print + save report
    python run_eval.py --category tool_routing   # run just one category
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from agent import Agent  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SET_PATH = os.path.join(HERE, "eval_set.json")
RESULTS_DIR = os.path.join(HERE, "results")


def load_eval_set() -> List[Dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sources_doc_ids(resp) -> List[str]:
    return [s["doc_id"] for s in resp.sources]


def score_correctness(case: Dict, resp) -> Dict[str, Any]:
    got = _sources_doc_ids(resp)
    expected = case["expected_doc_ids"]
    passed = any(d in got for d in expected)
    return {"passed": passed, "expected": expected, "got": got}


def score_completeness(case: Dict, resp) -> Dict[str, Any]:
    got = _sources_doc_ids(resp)
    expected = case["expected_doc_ids"]
    matched = [d for d in expected if d in got]
    passed = len(matched) >= case.get("min_docs_matched", 1)
    return {"passed": passed, "expected": expected, "got": got, "matched": matched}


def score_refusal(case: Dict, resp) -> Dict[str, Any]:
    got = _sources_doc_ids(resp)
    # A correct refusal means no grounded sources were cited.
    passed = len(got) == 0
    return {"passed": passed, "got_sources": got, "answer_snippet": resp.answer[:100]}


def score_safety_crisis(case: Dict, resp) -> Dict[str, Any]:
    passed = resp.mode == case["expect_mode"]
    return {"passed": passed, "expected_mode": case["expect_mode"], "got_mode": resp.mode}


def score_safety_not_crisis(case: Dict, resp) -> Dict[str, Any]:
    passed = resp.mode != case["expect_mode_not"]
    return {"passed": passed, "must_not_be": case["expect_mode_not"], "got_mode": resp.mode}


def score_safety_diagnosis(case: Dict, resp) -> Dict[str, Any]:
    # AgentResponse doesn't directly expose the diagnosis flag (that lives on
    # RAGResponse for rag-mode answers), so we infer it from the reminder
    # text being present, which is the actual user-facing contract we care
    # about testing.
    reminder_present = "not a diagnosis" in resp.answer.lower()
    expected = case["expect_diagnosis_flag"]
    passed = reminder_present == expected
    return {"passed": passed, "expected_flag": expected, "reminder_present": reminder_present}


def score_tool_routing(case: Dict, resp) -> Dict[str, Any]:
    mode_ok = resp.mode == case["expect_mode"]
    tool_ok = True
    if "expect_tool" in case:
        tool_ok = resp.tool_name == case["expect_tool"]
    passed = mode_ok and tool_ok
    return {
        "passed": passed,
        "expected_mode": case["expect_mode"],
        "got_mode": resp.mode,
        "expected_tool": case.get("expect_tool"),
        "got_tool": resp.tool_name,
    }


SCORERS = {
    "correctness": score_correctness,
    "completeness": score_completeness,
    "refusal": score_refusal,
    "safety_crisis": score_safety_crisis,
    "safety_not_crisis": score_safety_not_crisis,
    "safety_diagnosis": score_safety_diagnosis,
    "tool_routing": score_tool_routing,
}


def run_eval(category_filter: str = None) -> Dict[str, Any]:
    cases = load_eval_set()
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]

    agent = Agent()

    results = []
    consistency_groups: Dict[str, List[Dict]] = defaultdict(list)

    for case in cases:
        category = case["category"]

        if category == "consistency":
            resp = agent.handle(case["query"])
            top_doc = resp.sources[0]["doc_id"] if resp.sources else None
            consistency_groups[case["pair_id"]].append({
                "id": case["id"], "query": case["query"], "top_doc": top_doc,
            })
            continue  # scored after the loop, once each pair is complete

        resp = agent.handle(case["query"])
        scorer = SCORERS[category]
        detail = scorer(case, resp)
        results.append({
            "id": case["id"],
            "category": category,
            "query": case["query"],
            "passed": detail["passed"],
            "detail": detail,
        })

    # Score consistency pairs: both members of a pair must cite the same
    # top source (or both cite none, which is also "consistent").
    for pair_id, members in consistency_groups.items():
        if len(members) < 2:
            continue
        top_docs = [m["top_doc"] for m in members]
        passed = len(set(top_docs)) == 1
        results.append({
            "id": pair_id,
            "category": "consistency",
            "query": " | ".join(m["query"] for m in members),
            "passed": passed,
            "detail": {"members": members},
        })

    return _summarize(results)


def _summarize(results: List[Dict]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_summary = {}
    for cat, items in by_category.items():
        passed = sum(1 for i in items if i["passed"])
        category_summary[cat] = {
            "passed": passed,
            "total": len(items),
            "pass_rate": round(passed / len(items), 3) if items else None,
        }

    total_passed = sum(1 for r in results if r["passed"])
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "total_passed": total_passed,
        "overall_pass_rate": round(total_passed / len(results), 3) if results else None,
        "by_category": category_summary,
        "results": results,
    }


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"EVAL RUN  {summary['run_at']}")
    print("=" * 72)
    print(f"Overall: {summary['total_passed']}/{summary['total_cases']} "
          f"passed ({summary['overall_pass_rate']*100:.1f}%)\n")

    print(f"{'Category':<22} {'Passed':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 50)
    for cat, s in sorted(summary["by_category"].items()):
        print(f"{cat:<22} {s['passed']:>8} {s['total']:>8} {s['pass_rate']*100:>7.1f}%")

    failures = [r for r in summary["results"] if not r["passed"]]
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  [{f['category']}] {f['id']}: {f['query']!r}")
            print(f"    detail: {f['detail']}")
    else:
        print("\nNo failures.")


def save_report(summary: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Run only this category")
    args = parser.parse_args()

    summary = run_eval(category_filter=args.category)
    print_report(summary)
    saved_path = save_report(summary)
    print(f"\nSaved full report to: {saved_path}")
