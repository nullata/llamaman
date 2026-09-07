# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Regression guards for `load_state()`'s reattach path.

When llamaman restarts (to update, or on any process bounce) it walks its
saved instances and for each one whose container is still running, flips it
to `starting` so the poller re-verifies and lands it in `healthy`. Two
in-memory-only pieces have to be rebuilt in the same pass because they die
with the previous process:

  1. The **sidecar proxy** (werkzeug server on the public 8000-8020 port),
     which forwards to the container's internal port and gates the request
     with the RequestGate. Without a rebuild the public port sits unbound
     while the container keeps running, so a direct-port hit returns
     connection-refused.

  2. The **RequestGate** itself. `_public_instance` reads it via `get_gate`
     to emit the `queue` field, so without a rebuild the UI's
     "Queue N/M active · K queued" indicator disappears from the instance
     card, and the compat proxy on :42069 (which shares the same gate for
     hits routed by model name) stops enforcing max_concurrent.

Before the fix the block only ran for the sleeping restore path, so a
healthy instance whose container survived the restart came back missing
both pieces. These tests seed a JsonBackend with realistic saved rows,
mock docker probes to keep the reattach branch on-rails, run `load_state`,
and pin the invariant that the gate map and restore_proxies list are
populated for the reattached-running case too.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core.state import instances, downloads, load_state
from proxy import get_gate, remove_gate, _instance_gates, _shared_queue_gates
from storage.json_backend import JsonBackend


def _saved_instance(inst_id="i1", *, status="healthy", port=8000,
                    internal_port=9001, max_concurrent=3,
                    max_queue_depth=200, share_queue=False,
                    idle_timeout_min=0, proxy_sampling_override_enabled=False):
    """Build a saved-state row the way save_state() writes them."""
    config = {
        "max_concurrent": max_concurrent,
        "max_queue_depth": max_queue_depth,
        "share_queue": share_queue,
        "idle_timeout_min": idle_timeout_min,
        "proxy_sampling_override_enabled": proxy_sampling_override_enabled,
    }
    return {
        "id": inst_id,
        "model_name": "chat.gguf",
        "model_path": f"/models/{inst_id}.gguf",
        "port": port,
        "container_id": "cid-" + inst_id,
        "container_name": "llamaman-" + inst_id,
        "status": status,
        "log_file": "",
        "config": config,
        "started_at": 0,
        "llamaman_managed": True,
        "internal_port": internal_port,
        "stats": {},
    }


class ReattachedInstanceRebuildTests(unittest.TestCase):
    """After a llamaman restart, an instance whose container is still up
    must come back with its RequestGate and its sidecar-proxy restore entry
    populated - not just the sleeping-restore path (which was the only path
    the old code handled)."""

    def setUp(self):
        # Isolate the in-memory registries from other tests in the same run.
        instances.clear()
        downloads.clear()
        with patch("proxy._gates_lock"):
            pass  # touching the lock is enough to expose typo issues early
        # Actually clear the gate maps under the module's own lock.
        from proxy import _gates_lock
        with _gates_lock:
            _instance_gates.clear()
            _shared_queue_gates.clear()

        # A fresh per-test JsonBackend so seed rows don't leak across cases.
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"),
            os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"),
            os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        instances.clear()
        downloads.clear()
        from proxy import _gates_lock
        with _gates_lock:
            _instance_gates.clear()
            _shared_queue_gates.clear()
        self._tmp.cleanup()

    def _run_load_state(self, seed_rows,
                        container_running=True):
        """Seed the backend and run load_state under mocked docker probes.
        Returns whatever load_state returned (the restore_proxies list)."""
        self.storage.save_state(seed_rows, [], node_id="test-node")
        # core.state imports get_storage lazily inside load_state, so the
        # patch has to hit the storage module (where `from storage import
        # get_storage` resolves), not core.state's own namespace.
        with patch("storage.get_storage", return_value=self.storage), \
             patch("core.state.is_container_running",
                   return_value=container_running), \
             patch("core.state.stop_container"), \
             patch("core.state.resolve_llama_endpoint",
                   return_value=("localhost", 9001)), \
             patch("core.state.adopt_orphans", return_value=0):
            return load_state()

    def test_reattached_healthy_instance_rebuilds_gate(self):
        # A healthy instance with max_concurrent=3, container still up.
        # Pre-fix: no gate created, `queue` field missing from the card.
        seed = _saved_instance(status="healthy", max_concurrent=3,
                               max_queue_depth=200)
        self._run_load_state([seed])
        self.assertIn("i1", instances)
        self.assertEqual(instances["i1"]["status"], "starting")
        gate = get_gate("i1")
        self.assertIsNotNone(gate, "gate must be rebuilt on reattach")
        self.assertEqual(gate.max_concurrent, 3)
        self.assertEqual(gate.max_queue_depth, 200)

    def test_reattached_starting_instance_rebuilds_gate(self):
        # Saved status "starting" (mid-launch during the previous process)
        # follows the same reattach branch as "healthy" - gate rebuilt too.
        seed = _saved_instance(status="starting", max_concurrent=2)
        self._run_load_state([seed])
        gate = get_gate("i1")
        self.assertIsNotNone(gate)
        self.assertEqual(gate.max_concurrent, 2)

    def test_reattached_instance_with_proxy_added_to_restore_proxies(self):
        # The sidecar proxy (werkzeug server binding the public port) died
        # with the previous llamaman process. load_state signals app.py to
        # start_idle_proxy for it via the returned restore_proxies list.
        seed = _saved_instance(status="healthy", port=8000, internal_port=9001,
                               max_concurrent=3)
        restore_proxies = self._run_load_state([seed])
        self.assertIn(("i1", 8000, 9001), restore_proxies)

    def test_reattached_without_proxy_skips_restore_entry(self):
        # An instance launched without idle_timeout / max_concurrent /
        # proxy_sampling_override never had a sidecar in the first place -
        # docker publishes the container port directly to the public port.
        # No restore entry, no gate.
        seed = _saved_instance(status="healthy", internal_port=None,
                               max_concurrent=0, idle_timeout_min=0,
                               proxy_sampling_override_enabled=False)
        restore_proxies = self._run_load_state([seed])
        self.assertEqual(restore_proxies, [])
        self.assertIsNone(get_gate("i1"))

    def test_reattached_with_proxy_but_no_max_concurrent_rebuilds_proxy_only(self):
        # Idle-timeout-only instance: has a sidecar (for the wake-on-request
        # path) but no request-gating. Proxy restore fires, gate does not.
        seed = _saved_instance(status="healthy", max_concurrent=0,
                               idle_timeout_min=10)
        restore_proxies = self._run_load_state([seed])
        self.assertIn(("i1", 8000, 9001), restore_proxies)
        self.assertIsNone(get_gate("i1"))

    def test_stopped_instance_does_not_rebuild_gate_or_proxy(self):
        # An instance the user had explicitly stopped stays stopped on
        # restore; nothing about it goes on-air (no gate, no restore entry).
        seed = _saved_instance(status="stopped", max_concurrent=3)
        restore_proxies = self._run_load_state([seed])
        self.assertEqual(instances["i1"]["status"], "stopped")
        self.assertEqual(restore_proxies, [])
        self.assertIsNone(get_gate("i1"))

    def test_sleeping_restore_still_rebuilds_gate(self):
        # The pre-fix behavior for sleeping-with-proxy still holds - the
        # generalized block covers it identically. Container is gone here,
        # but has_proxy pins the restored status to "sleeping".
        seed = _saved_instance(status="sleeping", max_concurrent=4,
                               idle_timeout_min=5)
        restore_proxies = self._run_load_state([seed], container_running=False)
        self.assertEqual(instances["i1"]["status"], "sleeping")
        self.assertIn(("i1", 8000, 9001), restore_proxies)
        gate = get_gate("i1")
        self.assertIsNotNone(gate)
        self.assertEqual(gate.max_concurrent, 4)

    def test_reattach_starts_log_relay_with_follow_from_now(self):
        """After a llamaman restart, the reattach path must respawn the
        container-log relay daemon - the previous one died with the previous
        process, so /logs would return only stale content and /logs/stream
        would hang without new lines while the container is still writing.
        follow_from_now=True prevents replaying historical output already in
        the file. Regression guard for the missing relay-restart in load_state."""
        seed = _saved_instance(status="healthy", max_concurrent=3)
        seed["log_file"] = os.path.join(self._tmp.name, "i1.log")
        with patch("core.state.start_container_log_relay") as mock_relay:
            self._run_load_state([seed])
        mock_relay.assert_called_once()
        args, kwargs = mock_relay.call_args
        self.assertEqual(args[0], "cid-i1")  # container id passed through
        self.assertEqual(args[1], seed["log_file"])
        self.assertTrue(kwargs.get("follow_from_now"))

    def test_reattach_mints_log_path_when_saved_row_lacks_one(self):
        """Older state rows (or an adopted-then-saved instance from before
        the adopt path minted paths) have log_file="". Reattach must mint a
        LOGS_DIR/<inst_id>.log path so the /logs endpoints have something to
        read, instead of leaving the empty string that would break
        read_log_file / stream_log_file."""
        seed = _saved_instance(status="healthy", max_concurrent=0,
                               idle_timeout_min=0,
                               proxy_sampling_override_enabled=False)
        seed["log_file"] = ""
        with patch("core.state.start_container_log_relay") as mock_relay:
            self._run_load_state([seed])
        self.assertTrue(instances["i1"]["log_file"])
        self.assertTrue(instances["i1"]["log_file"].endswith("i1.log"))
        # Relay is still spawned with the minted path.
        args, _kwargs = mock_relay.call_args
        self.assertEqual(args[1], instances["i1"]["log_file"])

    def test_stopped_instance_does_not_start_log_relay(self):
        """A stopped instance stays stopped on restore; no container to tail,
        no relay to start. Guards against the reattach branch's log-relay
        call firing when it shouldn't."""
        seed = _saved_instance(status="stopped", max_concurrent=3)
        with patch("core.state.start_container_log_relay") as mock_relay:
            self._run_load_state([seed])
        mock_relay.assert_not_called()

    def test_share_queue_reattach_reuses_shared_gate(self):
        # Two share_queue peers of the same model must share ONE gate object
        # after restore, same invariant the launch path enforces via
        # _shared_queue_gates.
        seed_a = _saved_instance(inst_id="i1", port=8000, internal_port=9001,
                                 max_concurrent=2, share_queue=True)
        seed_b = _saved_instance(inst_id="i2", port=8001, internal_port=9002,
                                 max_concurrent=2, share_queue=True)
        # Both rows point at the same model_path so create_gate collapses them.
        seed_a["model_path"] = "/models/shared.gguf"
        seed_b["model_path"] = "/models/shared.gguf"
        self._run_load_state([seed_a, seed_b])
        g1 = get_gate("i1")
        g2 = get_gate("i2")
        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)
        self.assertIs(g1, g2, "share_queue peers must share one gate object")


if __name__ == "__main__":
    unittest.main()
