"""Tests for the evaluation core."""

from __future__ import annotations

import unittest
from pathlib import Path

from runner.eval import (
    calculate_average_score,
    load_cases,
    run_evaluation,
    score_response,
)
from runner.llm import MockLLMClient


class EvaluationCoreTest(unittest.TestCase):
    def test_contains_scoring_passes_when_expected_text_is_present(self) -> None:
        result = score_response("The answer is Paris.", "Paris", method="contains")

        self.assertEqual(result.score, 1)
        self.assertTrue(result.passed)

    def test_exact_scoring_normalizes_case_and_punctuation(self) -> None:
        result = score_response("Paris.", "paris", method="exact")

        self.assertEqual(result.score, 1)
        self.assertTrue(result.passed)

    def test_mock_runner_scores_sample_cases(self) -> None:
        cases = load_cases(Path("cases/cases.json"))
        results = run_evaluation(cases, MockLLMClient())

        self.assertEqual(calculate_average_score(results), 1.0)
        self.assertTrue(all(result.error is None for result in results))


if __name__ == "__main__":
    unittest.main()
