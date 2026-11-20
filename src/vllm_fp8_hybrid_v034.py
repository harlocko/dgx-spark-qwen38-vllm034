"""vllm_fp8_hybrid_modelopt — NVFP4 experts (ModelOpt) + blockwise-fp8 side layers.

The RadixArk checkpoint quantizes only the routed experts (NVFP4, ModelOpt) and
leaves the dense side layers (GDN in/out proj, QSA q/k/v/o, shared experts,
~15 GiB) in bf16. Those layers are read in full on every decoded token, so they
dominate decode bandwidth. This shim lets us store them as blockwise FP8-e4m3
(DeepSeek layout: fp8 ``weight`` + fp32 ``weight_scale_inv``, 128x128 blocks —
produced by the fork's ``tools/fp8_convert.py``) while keeping the NVFP4 MoE
path untouched.

Enabled by VLLM_FP8_HYBRID=1 (no-op otherwise). It patches
``ModelOptNvFp4Config``:
  * ``apply_vllm_mapper`` (called once with the model's hf->vllm name mapper):
    scans the checkpoint's safetensors metadata for F8_E4M3 ``.weight`` tensors
    that have a ``.weight_scale_inv`` sibling, and records their vLLM-side names;
  * ``get_quant_method``: for a LinearBase whose (possibly fused) sub-layers are
    all in that set, returns vLLM's generic blockwise ``Fp8Config`` method
    instead of the excluded/unquantized bf16 path.

Direct port of @Saren-Arterius's ``vllm_fp8_hybrid.py`` (AutoGPTQConfig) to the
ModelOpt NVFP4 config class.
"""
import logging
import os

logger = logging.getLogger("vllm.fp8_hybrid_modelopt")
_SENTINEL = "_fp8_hybrid_patched"


def _enabled() -> bool:
    return os.environ.get("VLLM_FP8_HYBRID", "0").lower() in ("1", "true", "yes")


def apply() -> None:
    """Enable checkpoint FP8 side projections on native ModelOpt mixed.

    Keep the native v0.34 dispatch for NVFP4 experts, MTP and PLE.
    """

    if not _enabled():
        return
    from vllm.model_executor.layers.quantization import modelopt as m

    cfg_cls = m.ModelOptMixedPrecisionConfig
    if not getattr(cfg_cls, _SENTINEL, False):
        _patch(cfg_cls)


def _patch(cfg_cls) -> None:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    orig_mapper = cfg_cls.apply_vllm_mapper
    orig_gqm = cfg_cls.get_quant_method

    def _scan_fp8_layers(self) -> set:
        try:
            from vllm.config import get_current_vllm_config
            model = get_current_vllm_config().model_config.model
            md = get_safetensors_params_metadata(model)
        except Exception as exc:
            raise RuntimeError(
                "v0.34 hybrid: cannot read safetensors metadata"
            ) from exc
        layers = {
            name[: -len(".weight")]
            for name, info in md.items()
            if name.endswith(".weight")
            and info.get("dtype") == "F8_E4M3"
            and name[: -len(".weight")] + ".weight_scale_inv" in md
            and any(
                part in name
                for part in (
                    ".linear_attn.",
                    ".self_attn.",
                    ".mlp.shared_expert.",
                )
            )
        }
        return layers

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        orig_mapper(self, hf_to_vllm_mapper)
        hf_layers = _scan_fp8_layers(self)
        self.fp8_layers = set(hf_to_vllm_mapper.apply_list(list(hf_layers))) if hf_layers else set()
        if self.fp8_layers:
            logger.info("fp8 hybrid (modelopt): %d blockwise-fp8 layers detected", len(self.fp8_layers))

    def _norm(name: str) -> str:
        """Compare on the part after 'layers.' so 'model.layers.3.x' == 'language_model.model.layers.3.x'."""
        i = name.find("layers.")
        return name[i:] if i >= 0 else name

    def _is_fp8_layer(self, prefix: str) -> bool:
        fp8_layers = getattr(self, "fp8_layers", None)
        if fp8_layers is None:
            fp8_layers = _scan_fp8_layers(self)
            self.fp8_layers = fp8_layers
        if not fp8_layers:
            return False
        norm_set = getattr(self, "_fp8_norm", None)
        if norm_set is None:
            norm_set = {_norm(l) for l in fp8_layers}
            self._fp8_norm = norm_set
        head, _, proj = prefix.rpartition(".")
        fused = (self.packed_modules_mapping or {}).get(proj)
        names = [f"{head}.{p}" for p in fused] if fused and head else [prefix]
        hit = all(_norm(n) in norm_set for n in names)
        if any(k in prefix for k in ("qkv_proj", "in_proj", "o_proj", "out_proj", "shared_expert")):
            stats = self.__dict__.setdefault("_fp8_stats", {"fp8": 0, "other": 0})
            stats["fp8" if hit else "other"] += 1
            if not hit and "layers.0." in prefix:
                logger.info("fp8 hybrid: NOT fp8: %s (checked %s)", prefix, [_norm(n) for n in names])
        return hit

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self._is_fp8_layer(prefix):
            fp8_cfg = getattr(self, "_fp8_cfg", None)
            if fp8_cfg is None:
                fp8_cfg = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=[128, 128],
                )
                fp8_cfg.packed_modules_mapping = self.packed_modules_mapping
                self._fp8_cfg = fp8_cfg
            st = self.__dict__.get("_fp8_stats")
            if st and st["fp8"] in (1, 50, 100, 150, 192, 200):
                logger.info("fp8 hybrid: %d fused modules dispatched to Fp8LinearMethod so far (e.g. %s)", st["fp8"], prefix)
            return fp8_cfg.get_quant_method(layer, prefix)
        return orig_gqm(self, layer, prefix)

    cfg_cls.apply_vllm_mapper = apply_vllm_mapper
    cfg_cls._is_fp8_layer = _is_fp8_layer
    cfg_cls.get_quant_method = get_quant_method
    setattr(cfg_cls, _SENTINEL, True)
    logger.info("fp8 hybrid patch applied to %s", cfg_cls.__name__)
