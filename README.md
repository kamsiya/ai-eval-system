# AI Eval System

A minimal, production-style AI evaluation system for testing LLM behavior against
a benchmark dataset.

The project loads evaluation cases, calls an LLM client, scores the model output
with rule-based binary graders, writes a structured JSON report, and prints a
summary score. The default LLM client is a deterministic mock, so the system runs
without an API key. For real model calls, use the OpenAI provider.

## Project Structure

```text
ai-eval-system/
  cases/cases.json
  model_compare.py
  runner/llm.py
  runner/eval.py
  main.py
  report/result.json
  report/model_compare.json
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
    "expected_output": "Paris",
    "scoring": "contains",
    "tags": ["qa", "geography"]
  }
]
```

Each case must include:

- `id`
- `input`
- `expected_output`

Optional fields:

- `scoring`: `exact` or `contains`
- `tags`: list of labels for slicing later

## How To Run

From the project directory:

```bash
python3 main.py
```

Optional CLI arguments:

```bash
python3 main.py --cases cases/cases.json --output report/result.json --provider mock
```

If your environment maps `python` to Python 3, this also works:

```bash
python main.py
```

## Run With OpenAI

Install dependencies and set your API key:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key"
```

Run a real model:

```bash
python3 main.py --provider openai --model gpt-5.5
```

You can also set a default model through the environment:

```bash
export OPENAI_MODEL="gpt-5.5"
python3 main.py --provider openai
```

## Compare Models

Compare several provider/model specs on the same cases:

```bash
python3 model_compare.py --models mock:gpt4o mock:claude mock:qwen mock:llama
```

Compare mock and OpenAI:

```bash
python3 model_compare.py --models mock:baseline openai:gpt-5.5
```

## Output Format

The report is saved to `report/result.json`:

```json
{
  "run": {
    "id": "20260706T120000Z",
    "created_at": "2026-07-06T12:00:00+00:00",
    "provider": "mock",
    "model": "mock-v1",
    "cases_path": "cases/cases.json"
  },
  "summary": {
    "total_cases": 3,
    "passed": 3,
    "failed": 0,
    "average_score": 1.0,
    "pass_rate": 1.0
  },
  "results": [
    {
      "id": "case_001",
      "input": "What is the capital of France?",
      "expected_output": "Paris",
      "actual_output": "Paris",
      "score": 1,
      "passed": true,
      "scoring": "contains",
      "reason": "expected text found in output",
      "provider": "mock",
      "model": "mock-v1",
      "latency_ms": 0.01,
      "error": null,
      "tags": ["qa", "geography"]
    }
  ]
}
```

## Extending The System

- Add more cases to `cases/cases.json`.
- Add a provider in `runner/llm.py` behind the `BaseLLMClient` interface.
- Add new scoring methods in `runner/eval.py`.
- Add an LLM-as-judge evaluator by introducing a new scorer function that returns
  the same `ScoreResult` shape.
- Use `model_compare.py` for regression checks across models or prompts.

## Tests

Run the built-in unit tests:

```bash
python3 -m unittest discover -s tests
```
