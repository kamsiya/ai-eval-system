# AI Eval System

A minimal, production-style AI evaluation system for testing LLM behavior against
a small benchmark dataset.

The project loads evaluation cases, calls an LLM client, scores the model output
with simple rule-based exact matching, writes a JSON report, and prints a summary
score. The default LLM client is a deterministic mock, so the system runs without
an API key.

## Project Structure

```text
ai-eval-system/
  cases/cases.json
  runner/llm.py
  runner/eval.py
  main.py
  report/result.json
  requirements.txt
  README.md
```

## Evaluation Cases

Cases are stored in `cases/cases.json`:

```json
[
  {
    "id": "case_001",
    "input": "What is the capital of France?",
    "expected_output": "Paris"
  }
]
```

Each case must include:

- `id`
- `input`
- `expected_output`

## How To Run

From the project directory:

```bash
python main.py
```

Optional CLI arguments:

```bash
python main.py --cases cases/cases.json --output report/result.json --provider mock
```

## Output Format

The report is saved to `report/result.json`:

```json
{
  "summary": {
    "total_cases": 3,
    "passed": 3,
    "average_score": 1.0
  },
  "results": [
    {
      "id": "case_001",
      "input": "What is the capital of France?",
      "expected_output": "Paris",
      "actual_output": "Paris",
      "score": 1
    }
  ]
}
```

## Extending The System

- Add more cases to `cases/cases.json`.
- Replace `MockLLMClient` in `runner/llm.py` with a real provider-backed client.
- Add new scoring functions in `runner/eval.py`.
- Add an LLM-as-judge evaluator later by keeping judge logic behind a separate
  scoring function or evaluator class.
