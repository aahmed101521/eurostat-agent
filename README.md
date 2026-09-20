# Eurostat Agent

**A provenance-first statistical AI research prototype for Eurostat.**

Eurostat Agent turns natural-language statistical questions into catalogue-grounded Eurostat queries and deterministic answers. Language models interpret questions and select from verified dataset candidates; ordinary Python code resolves dimensions, retrieves observations, performs arithmetic, and records where each result came from.

The central design principle is simple:

> **An answer without complete provenance is a failure.**

The implementation and its end-to-end robustness evaluation are complete as a **research prototype**. It is not presented as a production-ready assistant for arbitrary statistical questions.

## Why this project exists

A language model can produce a convincing statistical answer without having identified the right dataset, requested the right observation, or performed the calculation correctly. A source link alone does not resolve that problem.

Eurostat Agent separates interpretation from statistical execution. The model can propose *what to ask for*, but it cannot invent Eurostat dataset codes, dimension codes, observations, calculations, or provenance. The deterministic pipeline must verify and execute the request before an answer is returned.

## How it works

```text
Natural-language question
          |
          v
LLM interpretation -> QuestionPlan
          |
          v
Deterministic catalogue search
          |
          v
LLM selection from verified dataset candidates
          |
          v
StructuredQuestion and dimension/code resolution
          |
          v
Live Eurostat retrieval
          |
          v
Provenance validation and deterministic computation
          |
          v
DeterministicAnswer
          |
          +----> ExecutionTrace -> OperationalReport
          |
          v
Benchmarking and diagnostic evaluation
```

**Statistical provenance** records the source and computation behind an answer. **Execution tracing** records how the software ran, including failures, timings, model calls, and metadata-cache events. These are separate concerns.

### Implemented capabilities

- Deterministic Eurostat catalogue parsing and dataset search.
- Model-assisted question planning and selection **among verified catalogue candidates**.
- Dataset-specific dimension and code resolution.
- Live observation retrieval with statistical provenance.
- Deterministic `none`, `sum`, `difference`, `ratio`, and `percentage_change` computation primitives, where the request can be represented by the current question structure.
- End-to-end controller, structured benchmarks, and a calibrated qualitative judge.
- Local Ollama model/prompt experiments.
- Execution tracing, operational reporting, and execution-scoped structural metadata caching.
- A stratified end-to-end robustness benchmark with case-level evidence and an offline diagnostic audit.

Dataset freshness metadata and observations are retrieved live; statistical observations are **not** cached.

## Setup

The project was developed and tested with **Python 3.13.14** on Windows PowerShell. Install [Ollama](https://ollama.com/) separately to run the local-model experiments. Live execution also requires access to Eurostat's API.

```powershell
# Clone the repository
git clone https://github.com/aahmed101521/eurostat-agent.git
cd eurostat-agent

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project from the local source
python -m pip install -e .
```

The normal test suite does **not** require Ollama or live Eurostat access. The live benchmark does.

### Run the tests

```powershell
pytest -q
ruff check .
ruff format --check .
mypy src
pre-commit run --all-files
```

At the completed Stage 11 commit, the local test suite and GitHub CI passed **243 tests**. This is a snapshot of the evaluated repository state, not a promise about future commits.

## Running the end-to-end benchmark

The live robustness suite contains **36 questions** in six deliberately selected categories. Its default configurations use the constrained prompts for `llama3.2:3b` and `qwen3.5:latest`.

```powershell
# Install the local models if they are not already available
ollama pull llama3.2:3b
ollama pull qwen3.5:latest

# Inspect available cases and model configurations
python scripts\run_robustness_benchmark.py --list

# Run one case with one model before running the complete suite
python scripts\run_robustness_benchmark.py `
  --config llama32-3b-constrained `
  --case A01 `
  --output results\stage11_smoke.json

# Run the full 36-case suite for both default models
python scripts\run_robustness_benchmark.py
```

The full run writes its case-level JSON to `results/stage11_robustness.json`. Running it again can overwrite that file: preserve a copy if you want to compare separate runs. The `results/` directory is intentionally **untracked** and should not be committed by accident.

To generate retrospective diagnostic annotations from a saved run, without changing the original observations or calling either model:

```powershell
python scripts\audit_robustness_results.py
```

The companion report is written to `results/stage11_robustness_diagnostics.json`.

## Evaluation

The completed Stage 11 study ran **36 questions per model, 72 case executions in total**, using constrained local-model configurations. The benchmark distinguishes 20 supported questions, 10 known-unsupported questions, and 6 safe-failure questions per model.

| Recorded result | Llama 3.2 3B | Qwen 3.5 |
|---|---:|---:|
| Supported questions answered exactly | 11 / 20 (55%) | 13 / 20 (65%) |
| Cases labelled safe failure | 5 / 6 | 6 / 6 |
| Provenance violations recorded | **0** | **0** |
| Median end-to-end latency in this local run | 2.24 s | 514.10 s |

These figures describe **one local benchmark run**, not general model performance. The safe-failure label means that no answer was emitted; it does not necessarily mean that the model correctly recognized why a question was unsupported. Dataset/filter/operation accuracy of 100% in the run applies only to the **11 and 13 scored, supported, completed cases**, respectively—not all 36 prompts. The local latency figures should not be extrapolated to different hardware.

All 27 emitted answer objects carried recorded statistical-source and computation information, with **zero recorded provenance-violation flags**. This is evidence about the recorded run and the implemented checks, not independent verification of every source value or proof that each output answered the full original question.

The benchmark exposed an especially important distinction: **an answer can be provenance-complete while being request-incomplete**. In three known-unsupported, multi-value cases, Llama returned a verified observation for only one of the requested countries or sexes. The study also revealed dataset-selection failures, inconsistent temporal interpretation, and model-output-contract failures. Those limitations are retained as findings rather than concealed by the aggregate success rate.

For the full results and per-case analysis, see the [Stage 11 evidence audit](docs/stage11/stage11_evidence_audit.md). The raw experimental JSON is kept locally under `results/`, rather than versioned in this repository.

## Scope and limitations

Eurostat Agent is a **completed research prototype**, with a defined and evaluated scope:

- `QuestionPlan.filters` currently holds one string per dimension. Requests requiring several countries, years, ages, or sexes in the same dimension cannot be represented faithfully end to end.
- Annual periods are not silently normalized from full-date expressions such as `2024-01-01` to `2024`.
- Catalogue-grounded selection verifies that candidates exist; it does not guarantee that the model chose the dataset that answers the user's question.
- The available arithmetic primitives do not imply that every multi-observation computation can be expressed by the current planner.
- A visible failure is preferable to a plausible answer that the deterministic pipeline cannot verify.

The Stage 11 evaluation establishes a documented baseline for these limitations. Addressing them would constitute further research and development, **not a prerequisite for considering the present prototype complete**.

## Project structure

```text
src/eurostat_agent/         Core client, catalogue, controller, evaluation,
                            provenance, tracing, caching, and robustness modules
scripts/                    Local experiments and benchmark/audit runners
tests/                      Offline tests
docs/stage11/               Evidence audit and interpretation
results/                    Local experimental output (not versioned)
```

## Project status

**Research prototype complete through Stage 11.** The end-to-end robustness implementation and evidence audit were merged into `main` in [pull request #23](https://github.com/aahmed101521/eurostat-agent/pull/23), at commit `037d772`. The final recorded local quality gate passed **243 tests** and the GitHub pull-request CI check passed.

The core result of the project is an implemented and evaluated separation of responsibilities:

> **LLMs interpret. Deterministic code verifies, retrieves, computes, and records provenance. Evaluation measures behavior. Tracing observes execution.**
