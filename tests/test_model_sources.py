import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.downloads as downloads_api
import api.models as models_api
from core.model_sources import record_model_source
from core.state import downloads, downloads_lock


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        with downloads_lock:
            self._saved_downloads = {dl_id: dict(dl) for dl_id, dl in downloads.items()}
            downloads.clear()

    def tearDown(self):
        with downloads_lock:
            downloads.clear()
            downloads.update(self._saved_downloads)

    def test_record_model_source_persists_root_and_exact_model_path(self):
        storage = Mock()

        with patch("core.model_sources.get_storage", return_value=storage):
            record_model_source(
                "/models/Mistral-7B-Instruct-v0.3-Q4_K_M",
                "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                model_path="/models/Mistral-7B-Instruct-v0.3-Q4_K_M/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            )

        storage.merge_settings.assert_called_once_with({
            "model_sources": {
                os.path.realpath("/models/Mistral-7B-Instruct-v0.3-Q4_K_M"): {
                    "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                },
                os.path.realpath("/models/Mistral-7B-Instruct-v0.3-Q4_K_M/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"): {
                    "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                },
            }
        })

    def test_api_models_includes_repo_id_from_persisted_source_mapping(self):
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            nested_dir = os.path.join(models_dir, "download-root")
            os.makedirs(nested_dir, exist_ok=True)
            gguf_path = os.path.join(nested_dir, "model.gguf")
            with open(gguf_path, "wb") as f:
                f.write(b"gguf")

            storage = Mock()
            storage.get_settings.return_value = {
                "model_sources": {
                    nested_dir: {"repo_id": "org/model-repo"},
                }
            }

            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "get_storage", return_value=storage):
                resp = client.get("/api/models")

        self.assertEqual(resp.status_code, 200)
        models = resp.get_json()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["path"], gguf_path)
        self.assertEqual(models[0]["repo_id"], "org/model-repo")

    @patch("api.downloads.save_state")
    @patch("api.downloads.record_model_source")
    @patch("api.downloads._spawn_download_process")
    def test_download_create_records_model_source(self, spawn_mock, record_source_mock, _save_state_mock):
        app = Flask(__name__)
        app.register_blueprint(downloads_api.bp)
        client = app.test_client()

        proc = Mock(pid=4321)
        log_fh = Mock()
        with tempfile.TemporaryDirectory() as models_dir:
            spawn_mock.return_value = (proc, log_fh, "/tmp/dl-test.log")

            with patch.object(downloads_api, "MODELS_DIR", models_dir):
                resp = client.post("/api/downloads", json={
                    "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                    "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                })

            self.assertEqual(resp.status_code, 201)
            record_source_mock.assert_called_once_with(
                os.path.join(models_dir, "Mistral-7B-Instruct-v0.3-Q4_K_M"),
                "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                model_path=os.path.join(
                    models_dir,
                    "Mistral-7B-Instruct-v0.3-Q4_K_M",
                    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                ),
            )


if __name__ == "__main__":
    unittest.main()
