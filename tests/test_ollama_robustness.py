from eurostat_agent.ollama_robustness import (
    DEFAULT_ROBUSTNESS_CONFIG_NAMES,
    default_robustness_configs,
)


def test_primary_stage11_configs_are_only_constrained_local_models() -> None:
    configs = default_robustness_configs()

    assert tuple(config.name for config in configs) == DEFAULT_ROBUSTNESS_CONFIG_NAMES
    assert tuple(config.model for config in configs) == (
        "llama3.2:3b",
        "qwen3.5:latest",
    )
    assert all(config.prompt_variant == "constrained" for config in configs)
