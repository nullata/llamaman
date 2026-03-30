import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.instances as instances_api
from core.state import instances, instances_lock


class InstancesUiLaunchTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    def test_create_requires_ctx_size(self):
        resp = self.client.post(
            "/api/instances",
            json={
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "ctx_size": None,
            },
        )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "ctx_size is required")

    @patch("api.instances.launch_instance")
    def test_create_prompts_when_cap_reached_and_enforcement_disabled(self, launch_mock):
        storage = Mock()
        storage.get_settings.return_value = {"admin_ui_enforce_max_models": False}

        with patch("api.instances.get_storage", return_value=storage), \
             patch("api.instances.LLAMAMAN_MAX_MODELS", 1):
            with instances_lock:
                instances["chat"] = {
                    "id": "chat",
                    "model_name": "chat.gguf",
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "status": "healthy",
                    "started_at": 100,
                    "_last_request_at": 100,
                    "config": {"embedding_model": False},
                }

            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/new.gguf",
                    "port": 8001,
                    "ctx_size": 4096,
                },
            )

        self.assertEqual(resp.status_code, 409)
        data = resp.get_json()
        self.assertTrue(data["confirm_required"])
        self.assertIn("LLAMAMAN_MAX_MODELS=1", data["error"])
        launch_mock.assert_not_called()

    @patch("api.instances.stop_instance_by_id")
    @patch("api.instances.launch_instance")
    def test_create_evicts_when_cap_reached_and_enforcement_enabled(self, launch_mock, stop_mock):
        storage = Mock()
        storage.get_settings.return_value = {"admin_ui_enforce_max_models": True}
        launch_mock.return_value = ({
            "id": "new",
            "model_name": "new.gguf",
            "model_path": "/models/new.gguf",
            "port": 8001,
            "status": "starting",
            "config": {"embedding_model": False},
        }, None)

        with patch("api.instances.get_storage", return_value=storage), \
             patch("api.instances.LLAMAMAN_MAX_MODELS", 1):
            with instances_lock:
                instances["older"] = {
                    "id": "older",
                    "model_name": "older.gguf",
                    "model_path": "/models/older.gguf",
                    "port": 8000,
                    "status": "healthy",
                    "started_at": 100,
                    "_last_request_at": 100,
                    "config": {"embedding_model": False},
                }

            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/new.gguf",
                    "port": 8001,
                    "ctx_size": 4096,
                },
            )

        self.assertEqual(resp.status_code, 201)
        stop_mock.assert_called_once_with("older")
        launch_mock.assert_called_once()

    @patch("api.instances.stop_instance_by_id")
    @patch("api.instances.launch_instance")
    def test_create_allows_over_limit_after_confirmation_when_enforcement_disabled(self, launch_mock, stop_mock):
        storage = Mock()
        storage.get_settings.return_value = {"admin_ui_enforce_max_models": False}
        launch_mock.return_value = ({
            "id": "new",
            "model_name": "new.gguf",
            "model_path": "/models/new.gguf",
            "port": 8001,
            "status": "starting",
            "config": {"embedding_model": False},
        }, None)

        with patch("api.instances.get_storage", return_value=storage), \
             patch("api.instances.LLAMAMAN_MAX_MODELS", 1):
            with instances_lock:
                instances["chat"] = {
                    "id": "chat",
                    "model_name": "chat.gguf",
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "status": "healthy",
                    "started_at": 100,
                    "_last_request_at": 100,
                    "config": {"embedding_model": False},
                }

            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/new.gguf",
                    "port": 8001,
                    "ctx_size": 4096,
                    "confirm_overcommit": True,
                },
            )

        self.assertEqual(resp.status_code, 201)
        stop_mock.assert_not_called()
        launch_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
