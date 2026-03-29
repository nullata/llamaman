import os
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.llamaman as llamaman
from core.state import instances, instances_lock


class LlamamanPsTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(llamaman.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.llamaman._probe_server_ready", return_value=True)
    @patch(
        "api.llamaman.discover_models",
        return_value=[
            {
                "path": "/models/beta.gguf",
                "type": "gguf",
                "quant": "Q4_K_M",
                "size_bytes": 42,
            }
        ],
    )
    @patch(
        "api.llamaman.scan_llama_server_processes",
        return_value=[
            {
                "pid": 1234,
                "model_path": "/models/beta.gguf",
                "port": 9001,
                "config": {},
            }
        ],
    )
    def test_api_ps_includes_live_untracked_processes(
        self,
        _scan_mock,
        _discover_mock,
        _probe_mock,
    ):
        resp = self.client.get("/api/ps")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["models"]), 1)
        self.assertEqual(data["models"][0]["name"], "beta")
        self.assertEqual(data["models"][0]["size"], 42)
        self.assertEqual(data["models"][0]["details"]["quantization_level"], "Q4_K_M")

    @patch("api.llamaman.scan_llama_server_processes", return_value=[])
    @patch("api.llamaman._probe_server_ready", return_value=False)
    @patch("api.llamaman._instance_process_alive", return_value=True)
    @patch(
        "api.llamaman.discover_models",
        return_value=[
            {
                "path": "/models/alpha.gguf",
                "type": "gguf",
                "quant": "Q8_0",
                "size_bytes": 99,
            }
        ],
    )
    def test_api_ps_includes_live_tracked_process_with_stale_status(
        self,
        _discover_mock,
        _alive_mock,
        _probe_mock,
        _scan_mock,
    ):
        with instances_lock:
            instances["inst-1"] = {
                "id": "inst-1",
                "model_name": "alpha.gguf",
                "model_path": "/models/alpha.gguf",
                "port": 8000,
                "status": "stopped",
                "pid": 9999,
                "started_at": 1000,
                "config": {},
            }

        resp = self.client.get("/api/ps")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["models"]), 1)
        self.assertEqual(data["models"][0]["name"], "alpha")
        self.assertEqual(data["models"][0]["size"], 99)
        self.assertEqual(data["models"][0]["details"]["quantization_level"], "Q8_0")


if __name__ == "__main__":
    unittest.main()
