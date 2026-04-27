"""Plugin registry for CORAL-X experiment targets."""

from __future__ import annotations

from typing import Any

from core.common.config import CoralConfig
from core.ports.interfaces import Plugin


def _config_dict(config: CoralConfig) -> dict[str, Any]:
    """Return a plugin-friendly config dictionary."""
    return config.model_dump(mode="json")


def create_plugin(config: CoralConfig) -> Plugin:
    """Create the plugin for a supported experiment target."""
    target = config.experiment.target

    if target == "quixbugs_mini":
        from plugins.quixbugs_mini.plugin import QuixBugsMiniPlugin

        return QuixBugsMiniPlugin(_config_dict(config))

    if target == "fakenews_mini":
        from plugins.fakenews_mini.plugin import FakeNewsMiniPlugin

        return FakeNewsMiniPlugin(_config_dict(config))

    if target == "quixbugs_gemma4":
        from plugins.quixbugs_gemma4.plugin import QuixBugsGemma4Plugin

        return QuixBugsGemma4Plugin(_config_dict(config))

    if target == "ca_onemax":
        from plugins.ca_onemax.plugin import CAOneMaxPlugin

        return CAOneMaxPlugin(_config_dict(config))

    if target == "gsm8k_lora":
        from plugins.gsm8k_lora.plugin import GSM8KLoRAPlugin

        return GSM8KLoRAPlugin(_config_dict(config))

    supported = supported_targets()
    raise ValueError(
        f"Unsupported experiment target '{target}'. Supported targets: {', '.join(supported)}"
    )


def supported_targets() -> tuple[str, ...]:
    """Return supported plugin target names."""
    return (
        "quixbugs_mini",
        "fakenews_mini",
        "quixbugs_gemma4",
        "ca_onemax",
        "gsm8k_lora",
    )
