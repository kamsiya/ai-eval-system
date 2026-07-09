"""Compare multiple model/provider configurations on one benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from runner.eval import (
    SUPPORTED_SCORING_METHODS,
    build_report,
    load_cases,
    run_evaluation,
    save_report,
)
from runner.llm import create_llm_client


DEFAULT_CASES_PATH = Path("cases/cases.json")
DEFAULT_REPORT_PATH = Path("report/model_compare.json")
DEFAULT_MODELS = ["mock:gpt4o", "mock:claude", "mock:qwen", "mock:llama"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare models on one eval set.")
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
        help="Path where the comparison JSON report should be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to compare as provider:model specs. Examples: "
            "mock:gpt4o openai:gpt-5.5"
        ),
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


def parse_model_spec(spec: str) -> tuple[str, Optional[str]]:
    """Parse provider:model into provider and optional model."""

    if ":" not in spec:
        return spec.strip(), None

    provider, model = spec.split(":", 1)
    normalized_model = model.strip()
    return provider.strip(), normalized_model or None


def build_comparison_report(
    model_reports: list[dict[str, Any]], cases_path: Path
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    ranking = sorted(
        (
            {
                "provider": report["run"]["provider"],
                "model": report["run"]["model"],
                "average_score": report["summary"]["average_score"],
                "pass_rate": report["summary"]["pass_rate"],
            }
            for report in model_reports
        ),
        key=lambda item: (item["average_score"], item["pass_rate"]),
        reverse=True,
    )

    return {
        "run": {
            "id": created_at.strftime("%Y%m%dT%H%M%SZ"),
            "created_at": created_at.isoformat(),
            "cases_path": str(cases_path),
        },
        "ranking": ranking,
        "models": model_reports,
    }


def compare_models(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    model_reports: list[dict[str, Any]] = []

    for spec in args.models:
        provider, model = parse_model_spec(spec)
        llm_client = create_llm_client(
            provider=provider,
            model=model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        results = run_evaluation(
            cases=cases,
            llm_client=llm_client,
            scoring_override=args.scoring,
            fail_fast=args.fail_fast,
        )
        model_reports.append(
            build_report(
                results=results,
                cases_path=args.cases,
                provider=llm_client.provider,
                model=llm_client.model,
            )
        )

    return build_comparison_report(model_reports, args.cases)


def main() -> None:
    args = parse_args()
    report = compare_models(args)
    save_report(report, args.output)

    print("Model comparison:")
    for item in report["ranking"]:
        print(
            f"- {item['provider']}:{item['model']} "
            f"average={item['average_score']:.2f} "
            f"pass_rate={item['pass_rate']:.2%}"
        )
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
