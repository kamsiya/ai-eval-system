"""Run the AI evaluation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner.eval import build_report, load_cases, run_evaluation, save_report
from runner.llm import create_llm_client


DEFAULT_CASES_PATH = Path("cases/cases.json")
DEFAULT_REPORT_PATH = Path("report/result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM evaluation benchmark.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to evaluation cases JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the JSON report should be written.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM provider to use. Currently supported: mock.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_cases(args.cases)
    llm_client = create_llm_client(args.provider)
    results = run_evaluation(cases, llm_client)
    report = build_report(results)
    save_report(report, args.output)

    summary = report["summary"]
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed']}")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
