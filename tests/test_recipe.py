"""Offline checks for the published vLLM 0.34 recipe.

These tests do not start vLLM, allocate GPU memory or download model weights.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"


class SourceChecks(unittest.TestCase):
    def test_hook_modules_parse(self):
        for path in SRC.glob("*.py"):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding="utf-8"))

    def test_shell_scripts_parse(self):
        for path in SCRIPTS.glob("*.sh"):
            with self.subTest(file=path.name):
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dockerfile_references_existing_modules(self):
        for name in ("Dockerfile.v034-core", "Dockerfile.v034-yarn-mtp-v3"):
            dockerfile = (ROOT / name).read_text(encoding="utf-8")
            for line in dockerfile.splitlines():
                if line.strip().startswith("COPY src/"):
                    module = line.split()[1]
                    self.assertTrue((ROOT / module).exists(), module)

    def test_v034_base_is_pinned(self):
        source = (ROOT / "Dockerfile.v034-core").read_text(encoding="utf-8")
        self.assertIn("vllm/vllm-openai@sha256:", source)


class LauncherChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.fake_bin = self.base / "bin"
        self.fake_bin.mkdir()
        self.capture = self.base / "docker-arguments.txt"
        docker = self.fake_bin / "docker"
        docker.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then exit 0; fi\n"
            "if [ \"$1\" = container ] && [ \"$2\" = inspect ]; then\n"
            "  [ \"${FAKE_CONTAINER_EXISTS:-0}\" = 1 ] && exit 0 || exit 1\n"
            "fi\n"
            "if [ \"$1\" = run ]; then printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"; exit 0; fi\n"
            "echo 'Unexpected fake Docker command' >&2; exit 4\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        self.hf_cache = self.base / "hf"
        model = self.hf_cache / "hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4"
        (model / "refs").mkdir(parents=True)
        (model / "refs/main").write_text("abcdef\n", encoding="utf-8")
        snapshot = model / "snapshots/abcdef-fp8hybrid"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    def launch(self, exists=False):
        env = os.environ.copy()
        env.update(
            PATH=str(self.fake_bin) + os.pathsep + env.get("PATH", ""),
            HF_CACHE=str(self.hf_cache),
            CAPTURE_ARGS=str(self.capture),
            FAKE_CONTAINER_EXISTS="1" if exists else "0",
        )
        return subprocess.run(
            ["bash", str(SCRIPTS / "serve.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_serve_command_matches_inspected_runtime(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(args[0], "run")
        self.assertIn("--language-model-only", args)
        self.assertNotIn("--enforce-eager", args)
        self.assertEqual(args[args.index("--max-model-len") + 1], "500000")
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "1")
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "2048")
        self.assertEqual(args[args.index("--kv-cache-memory") + 1], "25769803776")
        self.assertEqual(args[args.index("-p") + 1], "127.0.0.1:18300:18300")
        self.assertIn("/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/abcdef-fp8hybrid", args)
        comp = json.loads(args[args.index("--compilation-config") + 1])
        spec = json.loads(args[args.index("--speculative-config") + 1])
        rope = json.loads(args[args.index("--hf-overrides") + 1])
        self.assertEqual(comp, {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 3]})
        self.assertEqual(spec, {"method": "mtp", "num_speculative_tokens": 2, "max_model_len": 500000})
        self.assertTrue(rope["text_config"]["rope_parameters"]["mrope_interleaved"])
        self.assertEqual(args[args.index("--engram-config") + 1], '{"disk_offload":true}')

    def test_existing_container_is_not_replaced(self):
        result = self.launch(exists=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
