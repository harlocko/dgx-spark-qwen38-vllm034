import ast
import importlib.util
import shutil
from pathlib import Path

source = Path("/opt/qwen38-v034/vllm_mtp_yarn_v034.py")

spec = importlib.util.find_spec("vllm")
assert spec is not None and spec.origin is not None

package = Path(spec.origin).resolve().parent
site = package.parent

target = package / "config" / "speculative.py"
destination = site / source.name
backup = target.with_name(
    target.name + ".orig-qwen38-v034-yarn"
)

marker = "# qwen38-v034-mtp-yarn-hook"

original = target.read_text()

assert "def compose_draft_hf_overrides(" in original
assert "Dict overrides are target-specific" in original
assert marker not in original
assert not destination.exists()
assert not backup.exists()

ast.parse(source.read_text())

hook = f"""

{marker}
from vllm_mtp_yarn_v034 import apply as _apply_qwen38_yarn
_apply_qwen38_yarn()
"""

ast.parse(original + hook)

shutil.copy2(target, backup)
shutil.copy2(source, destination)
target.write_text(original + hook)

print("MTP YaRN hook instalado")
print("Original:", backup)
print("Adaptador:", destination)
