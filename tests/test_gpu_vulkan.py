# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the Vulkan GPU vendor path (issue #85).

Two things have to hold together:
  1. `GPU_TYPE=vulkan` is an accepted vendor override that picks the
     server-vulkan image default and makes `_run_container` attach /dev/dri
     with the correct group_add - and does NOT attach /dev/kfd (that's a
     ROCm-only compute node; on a Vulkan-only host it does not exist).
  2. `group_add` is resolved to NUMERIC host GIDs whenever /dev/dri is
     visible from where llamaman runs, because Docker resolves group NAMES
     against the target container's /etc/group and the upstream server-vulkan
     image doesn't ship a `render` group - that mismatch is exactly what
     produced the original bug report:
        `Unable to find group render: no matching entries in group file`
     The fix applies to the rocm and intel branches too, not just vulkan,
     since the same name-resolution failure would bite any image that
     doesn't happen to preseed those names.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
import core.gpu as gpu


class ResolveRenderGidsTests(unittest.TestCase):
    """`resolve_render_gids` is the source of truth for the numeric GIDs we
    hand to Docker's group_add. It's a filesystem read of /dev/dri; the tests
    parametrize the directory so we don't depend on the CI host actually
    having a GPU. Whatever the host's real render/video GID happens to be is
    what we assert on."""

    def test_missing_directory_returns_empty(self):
        # The whole point of the empty-list return is "we don't know, fall
        # back to name-based group_add". If the /dev/dri dir doesn't exist
        # (bare-metal without a GPU, or a container that didn't get it
        # mounted), that has to surface as [] rather than an exception.
        self.assertEqual(gpu.resolve_render_gids("/nonexistent/dri"), [])

    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gpu.resolve_render_gids(tmp), [])

    def test_directory_with_nodes_returns_sorted_deduped_gids(self):
        # We can't chown to arbitrary GIDs without root, so we create real
        # files (they inherit the test-runner's GID) and assert that the
        # function returns EXACTLY that GID, once, sorted. That covers the
        # de-dup and sort contract without needing privileges.
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("renderD128", "renderD129", "card0", "card1"):
                open(os.path.join(tmp, name), "w").close()
            gids = gpu.resolve_render_gids(tmp)
            # All four files were created by the same user in the same
            # tempdir, so we expect exactly one GID and it should equal the
            # gid the test process sees on those files.
            expected = os.stat(os.path.join(tmp, "renderD128")).st_gid
            self.assertEqual(gids, [expected])

    def test_ignores_unrelated_entries(self):
        # /dev/dri also holds `by-path/` and other names; the globs are
        # narrow on purpose. A stray entry named `foo` must not show up in
        # the GID list.
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "renderD128"), "w").close()
            open(os.path.join(tmp, "foo"), "w").close()
            open(os.path.join(tmp, "by-path"), "w").close()
            gids = gpu.resolve_render_gids(tmp)
            # Should have exactly one entry (from renderD128), not three.
            self.assertEqual(len(gids), 1)


class GetVendorVulkanTests(unittest.TestCase):
    """`get_vendor()` reads GPU_TYPE from the env at every call (no cache on
    the override path), so `GPU_TYPE=vulkan` must round-trip through it
    verbatim. We deliberately do NOT auto-detect vulkan - a PCI vendor ID
    can't tell us whether the operator wants Vulkan over ROCm/SYCL - so
    there's no auto-detect test to write here."""

    def test_env_override_returns_vulkan(self):
        with patch.dict(os.environ, {"GPU_TYPE": "vulkan"}):
            self.assertEqual(gpu.get_vendor(), "vulkan")

    def test_env_override_is_case_insensitive_and_trimmed(self):
        # The auto-detect side of `get_vendor` lower-cases + strips the env
        # value; a preset saved by hand with capitals mustn't silently fall
        # through to auto-detection.
        with patch.dict(os.environ, {"GPU_TYPE": "  VULKAN  "}):
            self.assertEqual(gpu.get_vendor(), "vulkan")


class VulkanImageDefaultTests(unittest.TestCase):
    """The Vulkan image default lives in `_VENDOR_IMAGE_DEFAULTS`. If it
    disappears, `LLAMA_IMAGE` on a `GPU_TYPE=vulkan` install with no
    explicit `LLAMA_IMAGE` override falls back to `:server` (CPU), which
    would silently disable the GPU without any user-visible signal."""

    def test_vulkan_default_image_is_server_vulkan(self):
        import config
        self.assertEqual(
            config._VENDOR_IMAGE_DEFAULTS["vulkan"],
            "ghcr.io/ggml-org/llama.cpp:server-vulkan",
        )


class RunContainerVulkanBranchTests(unittest.TestCase):
    """`_run_container` builds the docker.containers.run(**kwargs) call. We
    mock the docker client and inspect kwargs directly - the invariants we
    care about are which devices get attached and what shape `group_add`
    takes for each vendor."""

    def _run(self, vendor: str, gids: list[int]):
        """Invoke `_run_container` with the given vendor + resolved GIDs and
        return the kwargs the mocked docker client was called with."""
        fake_container = Mock()
        fake_container.id = "abc123containerid"
        fake_client = Mock()
        fake_client.containers.run.return_value = fake_container

        with patch("api.instances.get_docker_client", return_value=fake_client), \
             patch("api.instances.ensure_docker_network"), \
             patch("api.instances._start_log_relay"), \
             patch("api.instances.get_vendor", return_value=vendor), \
             patch("core.gpu.resolve_render_gids", return_value=gids):
            container, err = instances_api._run_container(
                inst_id="inst-1",
                container_name="llamaman-inst-1",
                model_path="/models/chat.gguf",
                server_port=8000,
                config={"n_gpu_layers": -1, "ctx_size": 4096},
                log_file="/tmp/test.log",
            )
        self.assertIsNone(err)
        self.assertIs(container, fake_container)
        return fake_client.containers.run.call_args.kwargs

    def test_vulkan_branch_attaches_dri_only_not_kfd(self):
        # The whole point of the vulkan branch: /dev/dri is enough, and
        # /dev/kfd is a ROCm compute node that on a Vulkan-only host doesn't
        # exist. Attaching a missing device would fail the launch.
        kwargs = self._run("vulkan", gids=[44, 107])
        self.assertIn("devices", kwargs)
        self.assertEqual(kwargs["devices"], ["/dev/dri:/dev/dri"])
        for d in kwargs["devices"]:
            self.assertNotIn("/dev/kfd", d)

    def test_vulkan_branch_uses_numeric_group_add_from_stat(self):
        # Numeric GIDs, as strings, in sorted order. This is the fix for
        # issue #85: the container gets supplementary membership at 44/107
        # without any lookup against its own /etc/group.
        kwargs = self._run("vulkan", gids=[44, 107])
        self.assertEqual(kwargs["group_add"], ["44", "107"])

    def test_vulkan_branch_falls_back_to_names_when_dri_invisible(self):
        # If llamaman itself is in a container without /dev/dri mounted,
        # `resolve_render_gids` returns []. We MUST still populate group_add
        # - falling through to no groups means the container can't open
        # /dev/dri even though we mounted it. Names are the best we can do,
        # and preserve historical behaviour.
        kwargs = self._run("vulkan", gids=[])
        self.assertEqual(kwargs["group_add"], ["video", "render"])

    def test_vulkan_branch_does_not_set_device_requests(self):
        # `device_requests` triggers the NVIDIA Container Toolkit CDI path,
        # which errors on hosts without an NVIDIA runtime. It has no place
        # in the Vulkan branch.
        kwargs = self._run("vulkan", gids=[44, 107])
        self.assertNotIn("device_requests", kwargs)

    def test_rocm_branch_also_uses_numeric_group_add(self):
        # The name-resolution failure isn't Vulkan-specific - it's a
        # property of Docker's group_add semantics. Any llama.cpp image that
        # doesn't happen to define `render` in its /etc/group would hit the
        # same bug under the rocm branch, so the fix applies here too.
        kwargs = self._run("rocm", gids=[44, 107])
        self.assertEqual(kwargs["group_add"], ["44", "107"])
        # ROCm still attaches /dev/kfd - that's what distinguishes it from
        # the vulkan branch.
        self.assertIn("/dev/kfd:/dev/kfd", kwargs["devices"])

    def test_intel_branch_also_uses_numeric_group_add(self):
        kwargs = self._run("intel", gids=[44, 107])
        self.assertEqual(kwargs["group_add"], ["44", "107"])
        self.assertEqual(kwargs["devices"], ["/dev/dri:/dev/dri"])


class ResolveGroupAddHelperTests(unittest.TestCase):
    """`_resolve_group_add` is the boundary that bridges stat'd GIDs onto
    docker-py's group_add param. It has one job: numeric GIDs as strings
    when we have them, name fallback when we don't."""

    def test_returns_numeric_strings_when_gids_available(self):
        with patch("core.gpu.resolve_render_gids", return_value=[44, 107]):
            self.assertEqual(instances_api._resolve_group_add(), ["44", "107"])

    def test_returns_name_fallback_when_no_gids(self):
        with patch("core.gpu.resolve_render_gids", return_value=[]):
            self.assertEqual(instances_api._resolve_group_add(), ["video", "render"])


if __name__ == "__main__":
    unittest.main()
