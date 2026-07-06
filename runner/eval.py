"""Evaluation and reporting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.llm import BaseLLMClient


@dataclass(frozen=True)
class EvaluationCase:
    """A single benchmark item."""

    id: str
    input: str
    expected_output: str


@dataclass(frozen=True)
class EvaluationResult:
    """Result for a single benchmark item."""

    id: str
    input: str
    expected_output: str
    actual_output: str
    score: int


def normalize_text(text: str) -> str:
    """Normalize text for exact-match scoring."""

    return " ".join(text.strip().lower().split())


def score_response(actual_output: str, expected_output: str) -> int:
    """Return 1 for a normalized exact match, otherwise 0."""

    return int(normalize_text(actual_output) == normalize_text(expected_output))


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load benchmark cases from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    if not isinstance(raw_cases, list):
        raise ValueError("Cases file must contain a JSON array.")

    return [_parse_case(raw_case, index) for index, raw_case in enumerate(raw_cases)]


def run_evaluation(
    cases: list[EvaluationCase], llm_client: BaseLLMClient
) -> list[EvaluationResult]:
    """Run all evaluation cases against an LLM client."""

    results: list[EvaluationResult] = []

    for case in cases:
        actual_output = llm_client.generate(case.input)
        score = score_response(actual_output, case.expected_output)
        results.append(
            EvaluationResult(
                id=case.id,
                input=case.input,
                expected_output=case.expected_output,
                actual_output=actual_output,
                score=score,
            )
        )

    return results


def calculate_average_score(results: list[EvaluationResult]) -> float:
    """Calculate the average score across all results."""

    if not results:
        return 0.0

    return sum(result.score for result in results) / len(results)


def build_report(results: list[EvaluationResult]) -> dict[str, Any]:
    """Build a serializable report object."""

    return {
        "summary": {
            "total_cases": len(results),
            "passed": sum(result.score for result in results),
            "average_score": calculate_average_score(results),
        },
        "results": [
            {
                "id": result.id,
                "input": result.input,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "score": result.score,
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

    required_fields = ("id", "input", "expected_output")
    missing_fields = [field for field in required_fields if field not in raw_case]
    if missing_fields:
        raise ValueError(
            f"Case at index {index} is missing fields: {', '.join(missing_fields)}"
        )

    return EvaluationCase(
        id=str(raw_case["id"]),
        input=str(raw_case["input"]),
        expected_output=str(raw_case["expected_output"]),
    )
