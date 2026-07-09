"""Run the AI evaluation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner.eval import (
    SUPPORTED_SCORING_METHODS,
    build_report,
    load_cases,
    run_evaluation,
    save_report,
)
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
        help="LLM provider to use. Supported: mock, openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the selected provider. For OpenAI, defaults to OPENAI_MODEL or gpt-5.5.",
    )
    parser.add_argument(
        "--scoring",
        choices=sorted(SUPPORTED_SCORING_METHODS),
        default=None,
        help="Override the scoring method for all cases.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a model call or scoring step fails.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Provider request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Provider request retry count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_cases(args.cases)
    llm_client = create_llm_client(
        provider=args.provider,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    results = run_evaluation(
        cases=cases,
        llm_client=llm_client,
        scoring_override=args.scoring,
        fail_fast=args.fail_fast,
    )
    report = build_report(
        results=results,
        cases_path=args.cases,
        provider=llm_client.provider,
        model=llm_client.model,
    )
    save_report(report, args.output)

    summary = report["summary"]
    print(f"Provider: {llm_client.provider}")
    print(f"Model: {llm_client.model}")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
