"""Propagate RoPE-only dict overrides to the vLLM MTP draft."""

import functools
from collections.abc import Mapping
from copy import deepcopy

_MARKER = "_qwen38_v034_mtp_yarn_installed"
_KEYS = ("rope_parameters", "rope_theta")


def apply_rope_to_draft(
    updates,
    hf_config,
    transformers_mtp_overrides=None,
):
    from vllm.config.speculative import SpeculativeConfig
    from vllm.transformers_utils.config import get_hf_text_config

    config = deepcopy(hf_config)

    # vLLM first calls this override with a dummy config
    # to determine the draft model architecture.
    model_type = str(getattr(config, "model_type", ""))

    if model_type.startswith("dummy_"):
        return SpeculativeConfig.hf_config_override(
            config,
            transformers_mtp_overrides,
        )

    text = get_hf_text_config(config)

    if "rope_parameters" in updates:
        existing = deepcopy(getattr(text, "rope_parameters", None))
        incoming = deepcopy(updates["rope_parameters"])

        if not isinstance(existing, dict):
            raise TypeError("Existing rope_parameters is not a dict")
        if not isinstance(incoming, dict):
            raise TypeError("Override rope_parameters is not a dict")

        existing.update(incoming)
        text.rope_parameters = existing

    if "rope_theta" in updates:
        text.rope_theta = deepcopy(updates["rope_theta"])

    return SpeculativeConfig.hf_config_override(
        config,
        transformers_mtp_overrides,
    )


def apply():
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, _MARKER, False):
        return

    original = SpeculativeConfig.compose_draft_hf_overrides

    def compose(
        target_hf_overrides,
        transformers_mtp_overrides=None,
    ):
        if not isinstance(target_hf_overrides, Mapping):
            return original(
                target_hf_overrides,
                transformers_mtp_overrides,
            )

        selected = {
            k: deepcopy(target_hf_overrides[k])
            for k in _KEYS
            if k in target_hf_overrides
        }

        nested = target_hf_overrides.get("text_config")

        if isinstance(nested, Mapping):
            for key in _KEYS:
                if key in nested:
                    selected[key] = deepcopy(nested[key])

        if not selected:
            return original(
                target_hf_overrides,
                transformers_mtp_overrides,
            )

        return functools.partial(
            apply_rope_to_draft,
            selected,
            transformers_mtp_overrides=transformers_mtp_overrides,
        )

    SpeculativeConfig.compose_draft_hf_overrides = staticmethod(
        compose
    )

    setattr(SpeculativeConfig, _MARKER, True)
