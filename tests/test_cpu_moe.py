# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the MoE expert-offload launch control: build_llama_cmd sentinel
emission and preset persistence.

n_cpu_moe_layers is a single sentinel int fronting two llama.cpp flags:
    0   -> emit nothing (dense models default, and MoE-off)
    -1  -> emit --cpu-moe          (all layers' experts pinned to CPU)
    N>0 -> emit --n-cpu-moe N      (experts of the first N layers pinned to CPU)

These are shortcuts for tensor_buft_overrides against the LLM_FFN_EXPS_REGEX
(routed-expert FFN tensor names). Inert on dense models: llama.cpp silently
ignores overrides whose regex matches no tensor. See core/helpers.py."""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from api.presets import PRESET_HARDWARE_KEYS
from core.helpers import build_llama_cmd


class BuildLlamaCmdCpuMoeTests(unittest.TestCase):
    def test_missing_omits_both_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    def test_zero_omits_both_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"n_cpu_moe_layers": 0})
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    def test_negative_one_emits_cpu_moe_without_arg(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"n_cpu_moe_layers": -1})
        self.assertIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)
        # --cpu-moe takes no argument; the next token must be another flag or
        # end-of-list, never a bare number that llama-server would then try to
        # interpret as a positional.
        idx = cmd.index("--cpu-moe")
        after = cmd[idx + 1] if idx + 1 < len(cmd) else "--END"
        self.assertTrue(after.startswith("--"))

    def test_positive_n_emits_n_cpu_moe_with_arg(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"n_cpu_moe_layers": 8})
        self.assertIn("--n-cpu-moe", cmd)
        self.assertNotIn("--cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "8")

    def test_string_int_is_coerced(self):
        # Preset save currently stores ints, but the JSON boundary can hand us
        # a string via a hand-crafted request or a legacy row - int() coerces.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"n_cpu_moe_layers": "4"})
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "4")

    def test_none_treated_as_zero(self):
        # config.get() returning None (a nulled preset key) must NOT crash the
        # int() coercion; the `or 0` fallback catches it.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"n_cpu_moe_layers": None})
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)


class PresetSchemaCpuMoeTests(unittest.TestCase):
    def test_key_is_marked_per_node_hardware(self):
        # Scales with the node's VRAM, same tier as n_gpu_layers - a node with
        # more VRAM in the cluster can override to a smaller (or zero) value.
        self.assertIn("n_cpu_moe_layers", PRESET_HARDWARE_KEYS)


if __name__ == "__main__":
    unittest.main()
