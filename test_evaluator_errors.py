"""
Standalone Evaluator Test Runner — Deliberate Error Detection Test Suite
========================================================================
Runs deliberate test cases against the existing LangSmith Evaluator Agent
to verify that it accurately identifies correct answers, wrong outputs,
incorrect operators, incomplete answers, irrelevant statements, and misleading claims.

Usage:
    python test_evaluator_errors.py
"""

import io
import os
import sys
from typing import Any, Callable, Dict, List
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables
load_dotenv(override=True)

# Import evaluation function and schemas from existing app
try:
    from app import EvaluationResult, VerdictEnum, run_evaluation
except ImportError as e:
    print(f"❌ Error importing from app.py: {e}")
    sys.exit(1)


QUESTION = "What is the output of print(2 ** 3) in Python?"
REFERENCE_ANSWER = "The output is 8 because ** is the exponentiation operator in Python."


TEST_CASES = [
    {
        "name": "CORRECT",
        "sample_answer": "8",
        "expected_verdict": ["PASS"],
        "expected_error": False,
        "description": "Minimal correct direct answer",
        "validator": lambda res: (
            res.verdict.value == "PASS"
            and (
                res.detected_error_location.strip().lower() in ["none", "none.", "n/a", "no error"]
                or "no error" in res.detected_error_location.lower()
                or "accurate" in res.detected_error_location.lower()
            )
        ),
        "validation_criteria": "Verdict must be PASS and error location must be None / No error.",
    },
    {
        "name": "CORRECT_EXPLANATION",
        "sample_answer": "The output is 8 because ** is the exponentiation operator.",
        "expected_verdict": ["PASS"],
        "expected_error": False,
        "description": "Full correct answer with accurate explanation",
        "validator": lambda res: (
            res.verdict.value == "PASS"
            and (
                res.detected_error_location.strip().lower() in ["none", "none.", "n/a", "no error"]
                or "no error" in res.detected_error_location.lower()
                or "accurate" in res.detected_error_location.lower()
            )
        ),
        "validation_criteria": "Verdict must be PASS and error location must be None / No error.",
    },
    {
        "name": "WRONG_OUTPUT",
        "sample_answer": "The output is 6.",
        "expected_verdict": ["FAIL"],
        "expected_error": True,
        "description": "Incorrect computation output",
        "validator": lambda res: (
            res.verdict.value == "FAIL"
            and (
                "6" in res.detected_error_location
                or "6" in res.detected_error_explanation
                or "incorrect" in res.detected_error_explanation.lower()
                or "wrong" in res.detected_error_explanation.lower()
            )
        ),
        "validation_criteria": "Verdict must be FAIL and must identify the wrong output '6'.",
    },
    {
        "name": "WRONG_OPERATOR",
        "sample_answer": "The output is 6 because ** is the multiplication operator.",
        "expected_verdict": ["FAIL"],
        "expected_error": True,
        "description": "Incorrect output and false operator claim",
        "validator": lambda res: (
            res.verdict.value == "FAIL"
            and (
                "multiplication" in res.detected_error_location.lower()
                or "multiplication" in res.detected_error_explanation.lower()
                or "6" in res.detected_error_location
                or "6" in res.detected_error_explanation
                or "operator" in res.detected_error_explanation.lower()
            )
        ),
        "validation_criteria": "Verdict must be FAIL and identify the incorrect output or operator.",
    },
    {
        "name": "INCOMPLETE",
        "sample_answer": "The ** operator is used for exponentiation.",
        "expected_verdict": ["PARTIAL", "FAIL"],
        "expected_error": True,
        "description": "Explains the operator but omits the evaluated output",
        "validator": lambda res: (
            res.verdict.value in ["PARTIAL", "FAIL"]
            and (
                "missing" in res.detected_error_location.lower()
                or "output" in res.detected_error_location.lower()
                or "8" in res.detected_error_location
                or "missing" in res.detected_error_explanation.lower()
                or "output" in res.detected_error_explanation.lower()
                or "8" in res.detected_error_explanation
                or "does not provide" in res.detected_error_explanation.lower()
            )
        ),
        "validation_criteria": "Verdict must be PARTIAL or FAIL and identify that the output/value is missing.",
    },
    {
        "name": "IRRELEVANT",
        "sample_answer": "Python is a popular programming language.",
        "expected_verdict": ["FAIL"],
        "expected_error": True,
        "description": "Completely off-topic response",
        "validator": lambda res: (
            res.verdict.value == "FAIL"
            and (
                "entire answer" in res.detected_error_location.lower()
                or "irrelevant" in res.detected_error_explanation.lower()
                or "does not answer" in res.detected_error_explanation.lower()
                or "unrelated" in res.detected_error_explanation.lower()
                or "not address" in res.detected_error_explanation.lower()
            )
        ),
        "validation_criteria": "Verdict must be FAIL and identify that the answer is irrelevant or fails to address the question.",
    },
    {
        "name": "MISLEADING",
        "sample_answer": "The output is 8, but this expression causes a syntax error in modern Python.",
        "expected_verdict": ["FAIL", "PARTIAL"],
        "expected_error": True,
        "description": "Correct numerical result mixed with a false syntax error claim",
        "validator": lambda res: (
            res.verdict.value in ["FAIL", "PARTIAL"]
            and (
                "syntax error" in res.detected_error_location.lower()
                or "syntax error" in res.detected_error_explanation.lower()
                or "false" in res.detected_error_explanation.lower()
                or "misleading" in res.detected_error_explanation.lower()
                or "modern python" in res.detected_error_location.lower()
            )
        ),
        "validation_criteria": "Verdict must be FAIL/PARTIAL and identify the false claim regarding 'syntax error'.",
    },
]


def resolve_llm_configuration():
    """Detects available API keys and selects provider and model."""
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        provider = "Google Gemini"
        api_key = google_key
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    elif openai_key:
        provider = "OpenAI"
        api_key = openai_key
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        print("❌ Error: Neither GOOGLE_API_KEY nor OPENAI_API_KEY is configured in .env.")
        sys.exit(1)

    return provider, api_key, model_name


def run_all_tests():
    provider, api_key, model_name = resolve_llm_configuration()

    print("=" * 80)
    print("🧪 LangSmith Evaluator Agent — Deliberate Error Detection Test Runner")
    print("=" * 80)
    print(f"LLM Provider : {provider}")
    print(f"Model Name   : {model_name}")
    print(f"LangSmith    : {'Active (Tracing Enabled)' if os.getenv('LANGSMITH_API_KEY') else 'Disabled (No Key)'}")
    print(f"Question     : {QUESTION}")
    print(f"Reference    : {REFERENCE_ANSWER}")
    print("=" * 80)
    print()

    passed_count = 0
    total_tests = len(TEST_CASES)
    failed_details = []

    for idx, tc in enumerate(TEST_CASES, 1):
        name = tc["name"]
        sample = tc["sample_answer"]
        exp_verdicts = tc["expected_verdict"]
        tc_id = f"TEST-{idx:02d}_{name}"

        print(f"[{idx}/{total_tests}] Running Test: {name}")
        print(f"    Sample Answer   : \"{sample}\"")
        print(f"    Expected Verdict: {' / '.join(exp_verdicts)}")

        try:
            result: EvaluationResult = run_evaluation(
                question=QUESTION,
                reference_answer=REFERENCE_ANSWER,
                generated_answer=sample,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                test_case_id=tc_id,
            )

            actual_verdict = result.verdict.value
            det_loc = result.detected_error_location
            det_exp = result.detected_error_explanation
            has_error = det_loc.strip().lower() not in ["none", "none.", "n/a", "no error"] and "no error" not in det_loc.lower()

            is_test_passed = tc["validator"](result)

            print(f"    Actual Verdict  : {actual_verdict}")
            print(f"    Error Detected  : {'YES' if has_error else 'NO'}")
            print(f"    Error Location  : {det_loc}")
            print(f"    Explanation     : {det_exp}")

            if is_test_passed:
                print(f"    Result          : ✅ PASS")
                passed_count += 1
            else:
                print(f"    Result          : ❌ FAIL")
                failed_msg = (
                    f"Test '{name}' failed validation criteria: {tc['validation_criteria']}\n"
                    f"       Got Verdict: {actual_verdict}, Error Loc: '{det_loc}', Expl: '{det_exp}'"
                )
                failed_details.append(failed_msg)

        except Exception as e:
            err_msg = str(e)
            if api_key in err_msg:
                err_msg = err_msg.replace(api_key, "[REDACTED_KEY]")
            print(f"    Actual Verdict  : ERROR (Execution Failed: {err_msg})")
            print(f"    Result          : ❌ FAIL")
            failed_details.append(f"Test '{name}' threw exception: {err_msg}")

        print("-" * 80)

    print()
    print("=" * 80)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{passed_count}/{total_tests} evaluator tests passed.")

    if failed_details:
        print("\n⚠️ Failed Test Details:")
        for detail in failed_details:
            print(f"  • {detail}")
    else:
        print("\n🎉 All evaluator tests passed successfully with accurate error detection!")
    print("=" * 80)

    # Return exit code 0 if all tests pass, 1 if any fail
    return 0 if passed_count == total_tests else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
