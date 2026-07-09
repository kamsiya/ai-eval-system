"""Evaluation and reporting helpers."""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from runner.llm import BaseLLMClient

SUPPORTED_SCORING_METHODS = {"exact", "contains"}


@dataclass(frozen=True)
class EvaluationCase:
    """A single benchmark item."""

    id: str
    input: str
    expected_output: str
    scoring: str = "exact"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoreResult:
    """Rule-based score for a model response."""

    score: int
    passed: bool
    method: str
    reason: str


@dataclass(frozen=True)
class EvaluationResult:
    """Result for a single benchmark item."""

    id: str
    input: str
    expected_output: str
    actual_output: str
    score: int
    passed: bool
    scoring: str
    reason: str
    provider: str
    model: str
    latency_ms: float
    error: Optional[str] = None
    tags: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    """Normalize text for rule-based scoring."""

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def score_response(
    actual_output: str, expected_output: str, method: str = "exact"
) -> ScoreResult:
    """Return a binary rule-based score for a model response."""

    normalized_method = method.strip().lower()
    if normalized_method not in SUPPORTED_SCORING_METHODS:
        raise ValueError(
            f"Unsupported scoring method '{method}'. "
            f"Available methods: {', '.join(sorted(SUPPORTED_SCORING_METHODS))}"
        )

    actual = normalize_text(actual_output)
    expected = normalize_text(expected_output)

    if normalized_method == "exact":
        passed = actual == expected
        reason = "normalized exact match" if passed else "normalized exact mismatch"
    else:
        passed = expected in actual
        reason = (
            "expected text found in output"
            if passed
            else "expected text not found in output"
        )

    return ScoreResult(
        score=int(passed),
        passed=passed,
        method=normalized_method,
        reason=reason,
    )


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load benchmark cases from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    if not isinstance(raw_cases, list):
        raise ValueError("Cases file must contain a JSON array.")

    cases = [_parse_case(raw_case, index) for index, raw_case in enumerate(raw_cases)]
    case_ids = [case.id for case in cases]
    duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs found: {', '.join(duplicate_ids)}")

    return cases


def run_evaluation(
    cases: list[EvaluationCase],
    llm_client: BaseLLMClient,
    scoring_override: Optional[str] = None,
    fail_fast: bool = False,
) -> list[EvaluationResult]:
    """Run all evaluation cases against an LLM client."""

    results: list[EvaluationResult] = []
    provider = getattr(llm_client, "provider", llm_client.__class__.__name__)
    model = getattr(llm_client, "model", provider)

    for case in cases:
        started_at = time.perf_counter()
        scoring_method = scoring_override or case.scoring
        actual_output = ""
        error = None

        try:
            actual_output = llm_client.generate(case.input)
            score_result = score_response(
                actual_output=actual_output,
                expected_output=case.expected_output,
                method=scoring_method,
            )
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            if fail_fast:
                raise
            score_result = ScoreResult(
                score=0,
                passed=False,
                method=scoring_method,
                reason="LLM call or scoring failed",
            )

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        results.append(
            EvaluationResult(
                id=case.id,
                input=case.input,
                expected_output=case.expected_output,
                actual_output=actual_output,
                score=score_result.score,
                passed=score_result.passed,
                scoring=score_result.method,
                reason=score_result.reason,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                error=error,
                tags=case.tags,
            )
        )

    return results


def calculate_average_score(results: list[EvaluationResult]) -> float:
    """Calculate the average score across all results."""

    if not results:
        return 0.0

    return sum(result.score for result in results) / len(results)


def build_report(
    results: list[EvaluationResult],
    cases_path: Optional[Path] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Build a serializable report object."""

    created_at = datetime.now(timezone.utc)
    passed = sum(result.score for result in results)
    total_cases = len(results)
    resolved_provider = provider or (results[0].provider if results else None)
    resolved_model = model or (results[0].model if results else None)

    return {
        "run": {
            "id": created_at.strftime("%Y%m%dT%H%M%SZ"),
            "created_at": created_at.isoformat(),
            "provider": resolved_provider,
            "model": resolved_model,
            "cases_path": str(cases_path) if cases_path else None,
        },
        "summary": {
            "total_cases": total_cases,
            "passed": passed,
            "failed": total_cases - passed,
            "average_score": calculate_average_score(results),
            "pass_rate": calculate_average_score(results),
        },
        "results": [
            {
                "id": result.id,
                "input": result.input,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "score": result.score,
                "passed": result.passed,
                "scoring": result.scoring,
                "reason": result.reason,
                "provider": result.provider,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "tags": list(result.tags),
            }
            for result in results
        ],
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    """Save an evaluation report as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def _parse_case(raw_case: Any, index: int) -> EvaluationCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"Case at index {index} must be an object.")

    expected_output = raw_case.get("expected_output", raw_case.get("expected output"))
    required_fields = ("id", "input")
    missing_fields = [field for field in required_fields if field not in raw_case]
    if expected_output is None:
        missing_fields.append("expected_output")
    if missing_fields:
        raise ValueError(
            f"Case at index {index} is missing fields: {', '.join(missing_fields)}"
        )

    scoring = str(raw_case.get("scoring", "exact")).strip().lower()
    if scoring not in SUPPORTED_SCORING_METHODS:
        raise ValueError(
            f"Case {raw_case['id']} has unsupported scoring method '{scoring}'."
        )

    raw_tags = raw_case.get("tags", ())
    if isinstance(raw_tags, str):
        tags = (raw_tags,)
    elif isinstance(raw_tags, list):
        tags = tuple(str(tag) for tag in raw_tags)
    else:
        tags = ()

    return EvaluationCase(
        id=str(raw_case["id"]),
        input=str(raw_case["input"]),
        expected_output=str(expected_output),
        scoring=scoring,
        tags=tags,
    )
