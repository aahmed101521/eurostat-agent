import httpx
import pytest

from eurostat_agent.experiments import ExperimentConfig
from eurostat_agent.ollama_experiments import (
    CONSTRAINED_PLANNER_PROMPT_PREFIX,
    CONSTRAINED_SELECTOR_PROMPT_PREFIX,
    OLLAMA_EXPERIMENT_CONFIGS,
    PrefixedCompletionModel,
    build_ollama_planner_selector,
    select_benchmark_cases,
    select_ollama_configs,
)
from eurostat_agent.planner import (
    JsonDatasetSelector,
    JsonPlanner,
)


class FakeCompletionModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)

        return '{"ok":true}'


def test_ollama_experiment_matrix_has_two_models_and_two_prompts() -> None:
    assert tuple(
        (
            config.model,
            config.prompt_variant,
        )
        for config in (OLLAMA_EXPERIMENT_CONFIGS)
    ) == (
        (
            "llama3.2:3b",
            "baseline",
        ),
        (
            "llama3.2:3b",
            "constrained",
        ),
        (
            "qwen3.5:latest",
            "baseline",
        ),
        (
            "qwen3.5:latest",
            "constrained",
        ),
    )


def test_prefixed_completion_model_adds_constraints() -> None:
    inner = FakeCompletionModel()

    model = PrefixedCompletionModel(
        inner,
        "STRICT PREFIX",
    )

    result = model.complete("Original prompt")

    assert result == '{"ok":true}'

    assert inner.prompts == ["STRICT PREFIX\n\nOriginal prompt"]


def test_constrained_prompts_define_separate_contracts() -> None:
    assert "Return exactly one JSON object and nothing else" in (
        CONSTRAINED_PLANNER_PROMPT_PREFIX
    )

    assert (
        "Describe both the main statistical concept and any breakdown dimensions"
        in CONSTRAINED_PLANNER_PROMPT_PREFIX
    )

    assert (
        "Do not include observation-specific values"
        in CONSTRAINED_PLANNER_PROMPT_PREFIX
    )

    assert 'geography -> "geo"' in CONSTRAINED_PLANNER_PROMPT_PREFIX

    assert '"operation" must be exactly one of' in (CONSTRAINED_PLANNER_PROMPT_PREFIX)

    assert '"dataset_code": string' in (CONSTRAINED_SELECTOR_PROMPT_PREFIX)

    assert "Select the best dataset only from the candidate datasets" in (
        CONSTRAINED_SELECTOR_PROMPT_PREFIX
    )

    assert "Do not invent a dataset code" in (CONSTRAINED_SELECTOR_PROMPT_PREFIX)


def test_build_ollama_planner_selector_supports_baseline() -> None:
    config = ExperimentConfig(
        name="baseline",
        model="test-model",
        prompt_variant="baseline",
    )

    with httpx.Client() as http_client:
        planner, selector = build_ollama_planner_selector(
            config,
            http_client=http_client,
        )

    assert isinstance(
        planner,
        JsonPlanner,
    )

    assert isinstance(
        selector,
        JsonDatasetSelector,
    )


def test_build_ollama_planner_selector_rejects_unknown_variant() -> None:
    config = ExperimentConfig(
        name="unknown",
        model="test-model",
        prompt_variant="unknown",
    )

    with httpx.Client() as http_client:
        with pytest.raises(
            ValueError,
            match=("Unsupported prompt variant"),
        ):
            build_ollama_planner_selector(
                config,
                http_client=http_client,
            )


def test_select_ollama_configs_returns_requested_configuration() -> None:
    selected = select_ollama_configs(("llama32-3b-baseline",))

    assert len(selected) == 1

    assert selected[0].name == "llama32-3b-baseline"

    assert selected[0].model == "llama3.2:3b"


def test_select_ollama_configs_rejects_unknown_configuration() -> None:
    with pytest.raises(
        ValueError,
        match=("Unknown experiment configuration"),
    ):
        select_ollama_configs(("does-not-exist",))


def test_select_benchmark_cases_uses_one_based_indices() -> None:
    selected = select_benchmark_cases((1,))

    assert len(selected) == 1

    assert selected[0].question == (
        "What was the population of 20-year-old women in Belgium in 2024?"
    )


def test_select_benchmark_cases_rejects_invalid_index() -> None:
    with pytest.raises(
        ValueError,
        match=("Benchmark case index must be between 1 and 4"),
    ):
        select_benchmark_cases((0,))
