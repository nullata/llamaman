import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.instances as instances_api
import api.llamaman as llamaman_api
import api.presets as presets_api
from core.state import instances, instances_lock


class ProxySamplingTests(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    def test_preset_save_includes_proxy_sampling_fields(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        client = app.test_client()
        storage = Mock()

        with patch("api.presets.get_storage", return_value=storage):
            resp = client.put(
                "/api/presets/models/chat.gguf",
                json={
                    "ctx_size": 4096,
                    "proxy_sampling_override_enabled": True,
                    "proxy_sampling_temperature": 0.65,
                    "proxy_sampling_top_k": 25,
                    "proxy_sampling_top_p": 0.9,
                    "proxy_sampling_presence_penalty": 0.4,
                },
            )

        self.assertEqual(resp.status_code, 200)
        saved_model_path, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_model_path, "/models/chat.gguf")
        self.assertTrue(saved_preset["proxy_sampling_override_enabled"])
        self.assertEqual(saved_preset["proxy_sampling_temperature"], 0.65)
        self.assertEqual(saved_preset["proxy_sampling_top_k"], 25)
        self.assertEqual(saved_preset["proxy_sampling_top_p"], 0.9)
        self.assertEqual(saved_preset["proxy_sampling_presence_penalty"], 0.4)

    def test_preset_save_rejects_temperature_above_upper_bound(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        client = app.test_client()

        resp = client.put(
            "/api/presets/models/chat.gguf",
            json={
                "ctx_size": 4096,
                "proxy_sampling_override_enabled": True,
                "proxy_sampling_temperature": 2.5,
            },
        )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "proxy_sampling_temperature must be >= 0 and <= 2",
        )

    def test_preset_save_rejects_presence_penalty_above_upper_bound(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        client = app.test_client()

        resp = client.put(
            "/api/presets/models/chat.gguf",
            json={
                "ctx_size": 4096,
                "proxy_sampling_override_enabled": True,
                "proxy_sampling_presence_penalty": 2.5,
            },
        )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "proxy_sampling_presence_penalty must be >= -2 and <= 2",
        )

    @patch("api.instances.save_state")
    @patch("api.instances.start_idle_proxy")
    @patch("api.instances._run_container")
    @patch("api.instances.find_available_port", return_value=9001)
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_creates_proxy_when_sampling_override_enabled(
        self,
        _is_port_available_mock,
        _find_port_mock,
        run_container_mock,
        start_idle_proxy_mock,
        _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123containerid"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            proxy_sampling_override_enabled=True,
            proxy_sampling_temperature=0.55,
            proxy_sampling_top_k=17,
            proxy_sampling_top_p=0.88,
            proxy_sampling_presence_penalty=0.25,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(inst)
        self.assertEqual(inst["_internal_port"], 9001)
        self.assertTrue(inst["config"]["proxy_sampling_override_enabled"])
        self.assertEqual(inst["config"]["proxy_sampling_temperature"], 0.55)
        self.assertEqual(inst["config"]["proxy_sampling_top_k"], 17)
        self.assertEqual(inst["config"]["proxy_sampling_top_p"], 0.88)
        self.assertEqual(inst["config"]["proxy_sampling_presence_penalty"], 0.25)
        start_idle_proxy_mock.assert_called_once_with(inst["id"], 8000, 9001)

    @patch("api.llamaman.get_gate", return_value=None)
    @patch("api.llamaman._proxy_non_streaming")
    @patch("api.llamaman._ensure_model_running")
    def test_llamaman_chat_applies_proxy_sampling_overrides(
        self,
        ensure_model_running_mock,
        proxy_non_streaming_mock,
        _get_gate_mock,
    ):
        app = Flask(__name__)
        app.register_blueprint(llamaman_api.bp)
        client = app.test_client()

        ensure_model_running_mock.return_value = ({
            "id": "inst-1",
            "port": 8000,
            "status": "healthy",
            "config": {
                "proxy_sampling_override_enabled": True,
                "proxy_sampling_temperature": 0.3,
                "proxy_sampling_top_k": 7,
                "proxy_sampling_top_p": 0.72,
                "proxy_sampling_presence_penalty": -0.4,
            },
        }, None)
        proxy_non_streaming_mock.return_value = {
            "model": "chat",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }

        resp = client.post(
            "/api/chat",
            json={
                "model": "chat",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.1,
                "top_k": 99,
                "top_p": 0.12,
                "presence_penalty": 1.3,
            },
        )

        self.assertEqual(resp.status_code, 200)
        forwarded_body = proxy_non_streaming_mock.call_args.args[2]
        self.assertEqual(forwarded_body["temperature"], 0.3)
        self.assertEqual(forwarded_body["top_k"], 7)
        self.assertEqual(forwarded_body["top_p"], 0.72)
        self.assertEqual(forwarded_body["presence_penalty"], -0.4)


if __name__ == "__main__":
    unittest.main()
