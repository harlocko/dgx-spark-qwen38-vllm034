"""Install the Qwen3.8 FP8 compatibility hook into vLLM v0.34."""

import ast
import importlib.util
import shutil
from pathlib import Path

source = Path("/opt/qwen38-v034/vllm_fp8_hybrid_v034.py")

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise RuntimeError("vLLM package not found")

package = Path(spec.origin).resolve().parent
site_packages = package.parent

modelopt = (
    package
    / "model_executor"
    / "layers"
    / "quantization"
    / "modelopt.py"
)

destination = site_packages / "vllm_fp8_hybrid_v034.py"

if not source.is_file() or not modelopt.is_file():
    raise RuntimeError("Required source files not found")

if destination.exists():
    raise RuntimeError("FP8 compatibility module already installed")

shim = source.read_text()
ast.parse(shim)

original = modelopt.read_text()
tree = ast.parse(original)

classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "ModelOptMixedPrecisionConfig"
]

if len(classes) != 1:
    raise RuntimeError("Unexpected ModelOptMixedPrecisionConfig API")

methods = {
    node.name for node in classes[0].body
    if isinstance(node, ast.FunctionDef)
}

required = {"apply_vllm_mapper", "get_quant_method"}

if not required.issubset(methods):
    raise RuntimeError(
        f"ModelOpt API changed: missing {required - methods}"
    )

marker = "# qwen38-v034-fp8-hybrid-hook"

if marker in original:
    raise RuntimeError("FP8 hook already present")

hook = f"""

{marker}
from vllm_fp8_hybrid_v034 import apply as _qwen38_fp8_apply
_qwen38_fp8_apply()
"""

modified = original + hook
ast.parse(modified)

backup = modelopt.with_name(
    modelopt.name + ".orig-qwen38-v034"
)

shutil.copy2(modelopt, backup)
shutil.copy2(source, destination)
modelopt.write_text(modified)

print("vLLM package:", package)
print("FP8 module:", destination)
print("ModelOpt hook:", modelopt)
print("Original backup:", backup)
print("Static compatibility checks: OK")
