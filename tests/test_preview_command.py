# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for POST /api/preview-command (issue #90).

The endpoint's whole job is to render the SAME shell-escaped llama-server
command that a Launch with the given form body would actually produce. It
does this by feeding the body straight into build_llama_cmd - if these two
paths ever diverge, the "Copy llama.cpp command" button in the UI would
show something the operator can't reproduce, which defeats its point.

We test:
  1. The happy path: a config renders the expected flag list.
  2. Shell escaping: paths with spaces come back single-quoted (shlex.join).
  3. Every returned string starts with "llama-server ".
  4. model_path is required (empty / missing -> 400).
  5. build_llama_cmd's flag emission still governs the output when non-obvious
     defaults are in play - specifically, empty means auto (no flag emitted).
  6. The FLAG SET matches what build_llama_cmd itself emits for the same
     config (not just a spot check). This is the drift trip-wire: if
     build_llama_cmd starts (or stops) emitting a flag, the preview endpoint
     picks it up automatically, and this test proves it.
"""

import os
import shlex
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from flask import Flask

import api.instances as instances_api
from core.helpers import build_llama_cmd


class PreviewCommandTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(instances_api.bp)
        self.client = app.test_client()

    def _post(self, body):
        return self.client.post("/api/preview-command", json=body)

    def test_missing_model_path_returns_400(self):
        # An empty body is a common mistake (calling from a form with no
        # model selected). Rendering a command for "" would put --model ''
        # into the CLI which is nonsense; refuse it up front so the UI shows
        # a clear error instead of copying a broken command.
        r = self._post({})
        self.assertEqual(r.status_code, 400)
        self.assertIn("model_path", r.get_json().get("error", ""))

    def test_blank_model_path_returns_400(self):
        r = self._post({"model_path": "   "})
        self.assertEqual(r.status_code, 400)

    def test_happy_path_starts_with_llama_server_and_has_model_flag(self):
        r = self._post({
            "model_path": "/models/chat.gguf",
            "port": 8000,
            "ctx_size": 4096,
        })
        self.assertEqual(r.status_code, 200)
        cmd = r.get_json()["command"]
        self.assertTrue(cmd.startswith("llama-server "))
        # shlex.split round-trip is the canonical way to inspect a joined
        # command line - protects the test from cosmetic quoting drift.
        tokens = shlex.split(cmd)
        self.assertEqual(tokens[0], "llama-server")
        self.assertIn("--model", tokens)
        self.assertEqual(tokens[tokens.index("--model") + 1], "/models/chat.gguf")
        self.assertIn("--port", tokens)
        self.assertEqual(tokens[tokens.index("--port") + 1], "8000")

    def test_paths_with_spaces_are_shell_escaped(self):
        # If a user names their model dir with a space (or a hyphen followed
        # by "$SHELL_VAR" trickery), shlex.join has to quote it so the
        # command, when pasted into a real shell, refers to the same file
        # llama-server would open here. Without escaping the copied command
        # would blow up in bash with "no such file".
        r = self._post({
            "model_path": "/models/my models/chat.gguf",
            "ctx_size": 4096,
        })
        self.assertEqual(r.status_code, 200)
        cmd = r.get_json()["command"]
        # shlex.split MUST give us back the exact path when we round-trip.
        tokens = shlex.split(cmd)
        self.assertEqual(tokens[tokens.index("--model") + 1],
                         "/models/my models/chat.gguf")
        # And it should not just be inlined with a raw space - that would
        # mean shell splitting sees two args instead of one.
        self.assertIn("'/models/my models/chat.gguf'", cmd)

    def test_flag_set_matches_build_llama_cmd_exactly(self):
        # This is the drift trip-wire. If build_llama_cmd starts / stops
        # emitting a flag for a given config, the preview endpoint MUST
        # follow, and this test proves that by comparing token-by-token
        # against build_llama_cmd's own output for the same config. If
        # these two paths ever have to diverge, this test forces the
        # divergence to be an intentional edit rather than a silent bug.
        config = {
            "model_path": "/models/chat.gguf",
            "port": 9000,
            "ctx_size": 4096,
            "n_gpu_layers": 32,
            "flash_attn": "on",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "reasoning_format": "deepseek",
            "load_mode": "mmap+mlock",
            "split_mode": "row",
            "tensor_split": "24,16",
            "threads": 8,
            "extra_args": "--verbose",
        }
        r = self._post(config)
        self.assertEqual(r.status_code, 200)
        cmd_tokens = shlex.split(r.get_json()["command"])
        expected_tokens = ["llama-server"] + build_llama_cmd(
            config["model_path"], config["port"], config,
        )
        self.assertEqual(cmd_tokens, expected_tokens)

    def test_empty_means_auto_omits_flag(self):
        # build_llama_cmd's "empty means llama.cpp default" contract is what
        # keeps the CLI quiet for the common case - preview must reflect it.
        # Passing flash_attn='auto' should mean --flash-attn is NOT emitted
        # (auto is the default and llama.cpp needs the flag only for on/off).
        r = self._post({
            "model_path": "/models/chat.gguf",
            "ctx_size": 4096,
            "flash_attn": "auto",
        })
        self.assertEqual(r.status_code, 200)
        tokens = shlex.split(r.get_json()["command"])
        self.assertNotIn("--flash-attn", tokens)

    def test_port_defaults_to_8000_when_absent_or_bad(self):
        # Missing port -> 8000. Non-int port (e.g. a hand-crafted body) also
        # -> 8000, rather than 500-ing the request. The user's real launch
        # picks a port at the API layer above, so preview being tolerant
        # here matches the "reflect the form, don't police it" contract.
        for body in ({"model_path": "/models/chat.gguf", "ctx_size": 4096},
                     {"model_path": "/models/chat.gguf", "ctx_size": 4096, "port": "banana"}):
            r = self._post(body)
            self.assertEqual(r.status_code, 200)
            tokens = shlex.split(r.get_json()["command"])
            self.assertEqual(tokens[tokens.index("--port") + 1], "8000")


if __name__ == "__main__":
    unittest.main()
