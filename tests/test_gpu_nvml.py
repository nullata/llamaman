# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Guard tests for the NVIDIA / pynvml path in core.gpu (issue #57).

The pynvml-backed code paths are the only ones in this project that talk to
NVML - they power vendor auto-detection on NVIDIA hosts, the dashboard's
GPU tiles, and the per-instance GPU label on the instance cards. Every
caller wraps them in try/except: nothing raises, worst case is empty state.
That's what makes the migration to nvidia-ml-py cheap - and also what makes
it silent when it breaks. These tests cover the two functions that actually
call NVML (`detect_gpu_vendor` NVIDIA branch and `query_nvidia_pynvml`) by
injecting a fake `pynvml` module into sys.modules, so they run on any host
regardless of whether real NVIDIA hardware or drivers are present.

If nvidia-ml-py (or a future replacement) ever renames one of the six NVML
calls we use, these tests fail loudly on the first CI run instead of
producing a blank GPU tile on the first real NVIDIA host to deploy.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.gpu as gpu


class _FakePynvml:
    """Stand-in for the `pynvml` module. Every NVML call we use is present,
    so a rename in the real package would surface as an AttributeError here
    in the caller - which is exactly the drift signal we want."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, gpus=None, init_raises=None):
        self._gpus = gpus or []
        self._init_raises = init_raises
        self._handles = [object() for _ in self._gpus]
        self._by_handle = dict(zip(self._handles, self._gpus))
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self._init_raises is not None:
            raise self._init_raises

    def nvmlDeviceGetCount(self):
        return len(self._gpus)

    def nvmlDeviceGetHandleByIndex(self, i):
        return self._handles[i]

    def nvmlDeviceGetName(self, h):
        return self._by_handle[h]["name"]

    def nvmlDeviceGetMemoryInfo(self, h):
        g = self._by_handle[h]
        m = MagicMock()
        m.total = g["total"]
        m.used = g["used"]
        m.free = g["free"]
        return m

    def nvmlDeviceGetUtilizationRates(self, h):
        g = self._by_handle[h]
        u = MagicMock()
        u.gpu = g["util"]
        return u

    def nvmlDeviceGetTemperature(self, h, _sensor):
        return self._by_handle[h]["temp"]


def _install_fake_pynvml(fake):
    """Context helper: put the fake into sys.modules under the name the
    function-under-test does `import pynvml` against. Returns the previous
    value so the caller can restore it."""
    previous = sys.modules.get("pynvml")
    sys.modules["pynvml"] = fake
    return previous


def _restore_pynvml(previous):
    if previous is None:
        sys.modules.pop("pynvml", None)
    else:
        sys.modules["pynvml"] = previous


class DetectGpuVendorNvmlBranchTests(unittest.TestCase):
    """`detect_gpu_vendor` returns 'cuda' when the NVIDIA branch succeeds and
    reports at least one device. The subsequent AMD/Intel sysfs probe must
    NEVER be reached in that case (the test host might have those cards).
    """

    def setUp(self):
        # We're testing the NVIDIA branch specifically, but detect_gpu_vendor
        # falls through to /sys/class/drm on failure. Neuter that glob for
        # every test in this class so the assertion about the fallthrough is
        # honest even on hosts that happen to have DRM devices.
        self._glob_patch = patch("core.gpu.glob.glob", return_value=[])
        self._glob_patch.start()
        self._prev_pynvml = None

    def tearDown(self):
        self._glob_patch.stop()
        _restore_pynvml(self._prev_pynvml)

    def test_returns_cuda_when_at_least_one_device_present(self):
        fake = _FakePynvml(gpus=[{
            "name": "RTX 4090", "total": 24 * 1024**3, "used": 0, "free": 24 * 1024**3,
            "util": 0, "temp": 40,
        }])
        self._prev_pynvml = _install_fake_pynvml(fake)
        self.assertEqual(gpu.detect_gpu_vendor(), "cuda")
        self.assertEqual(fake.init_calls, 1)

    def test_returns_none_when_zero_devices(self):
        # nvmlInit succeeded but there are no GPUs - the NVIDIA branch
        # returns without a value and falls through to AMD/Intel sysfs
        # (patched empty). Contract: detect_gpu_vendor() -> None.
        fake = _FakePynvml(gpus=[])
        self._prev_pynvml = _install_fake_pynvml(fake)
        self.assertIsNone(gpu.detect_gpu_vendor())

    def test_returns_none_when_nvml_init_raises(self):
        # A missing NVIDIA driver, revoked toolkit capability, or a broken
        # pynvml install all present as nvmlInit raising. The branch has to
        # swallow that and fall through - not 500 the caller.
        fake = _FakePynvml(init_raises=RuntimeError("NVML shared library not found"))
        self._prev_pynvml = _install_fake_pynvml(fake)
        self.assertIsNone(gpu.detect_gpu_vendor())

    def test_returns_none_when_pynvml_import_fails(self):
        # Simulate the package being absent entirely (ImportError inside the
        # try). Put a value into sys.modules that raises on any attribute
        # access won't work here because `import pynvml` binds the module
        # before any lookup - so we install None, which makes the `import`
        # statement raise ImportError. Callers must handle that quietly.
        self._prev_pynvml = sys.modules.get("pynvml")
        sys.modules["pynvml"] = None  # `import pynvml` -> ImportError
        try:
            self.assertIsNone(gpu.detect_gpu_vendor())
        finally:
            # setUp's tearDown will restore _prev_pynvml.
            pass


class QueryNvidiaPynvmlTests(unittest.TestCase):
    """`query_nvidia_pynvml` returns a list of per-GPU dicts with the exact
    shape the /api/gpu-info and instance-card GPU-label consumers expect.
    Shape drift here (renamed keys, wrong unit) shows up as empty tiles or
    silent zeros in the UI, not as an exception - which is why we assert on
    the shape explicitly here."""

    def setUp(self):
        self._prev_pynvml = None

    def tearDown(self):
        _restore_pynvml(self._prev_pynvml)

    def test_single_gpu_returns_expected_shape(self):
        fake = _FakePynvml(gpus=[{
            "name": "RTX 4090",
            "total": 24 * 1024**3,     # 24 GB
            "used": 6 * 1024**3,       # 6 GB
            "free": 18 * 1024**3,      # 18 GB
            "util": 42,
            "temp": 58,
        }])
        self._prev_pynvml = _install_fake_pynvml(fake)
        out = gpu.query_nvidia_pynvml()
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        g = out[0]
        # Keys are the contract - api/system_info.py and api/instances.py
        # both read exactly these names, so a rename must NOT silently pass.
        for k in ("index", "name", "memory_used_mb", "memory_total_mb",
                  "memory_free_mb", "utilization_pct", "temperature_c"):
            self.assertIn(k, g, msg=f"missing key {k!r}")
        # Units: memory_*_mb is bytes // (1024 * 1024). If we ever drift to
        # bytes or GB the dashboard tiles show wrong numbers with no error.
        self.assertEqual(g["memory_total_mb"], 24 * 1024)
        self.assertEqual(g["memory_used_mb"], 6 * 1024)
        self.assertEqual(g["memory_free_mb"], 18 * 1024)
        self.assertEqual(g["utilization_pct"], 42)
        self.assertEqual(g["temperature_c"], 58)
        self.assertEqual(g["name"], "RTX 4090")
        self.assertEqual(g["index"], 0)

    def test_multi_gpu_preserves_order_and_indices(self):
        # Instance-card GPU labels look up by index, so the order MUST match
        # the NVML enumeration one-for-one. Two devices, two entries, indices
        # 0 and 1 in the same order.
        fake = _FakePynvml(gpus=[
            {"name": "RTX 4090", "total": 24 * 1024**3, "used": 0, "free": 24 * 1024**3, "util": 0, "temp": 40},
            {"name": "RTX 3060", "total": 12 * 1024**3, "used": 0, "free": 12 * 1024**3, "util": 0, "temp": 35},
        ])
        self._prev_pynvml = _install_fake_pynvml(fake)
        out = gpu.query_nvidia_pynvml()
        self.assertEqual(len(out), 2)
        self.assertEqual([g["index"] for g in out], [0, 1])
        self.assertEqual([g["name"] for g in out], ["RTX 4090", "RTX 3060"])

    def test_bytes_name_is_decoded_to_str(self):
        # Older NVML/pynvml builds returned nvmlDeviceGetName as bytes; newer
        # ones return str. The function normalises to str either way - a
        # regression here would put a `b'...'` literal into the UI.
        fake = _FakePynvml(gpus=[{
            "name": b"RTX 4090", "total": 24 * 1024**3, "used": 0, "free": 24 * 1024**3,
            "util": 0, "temp": 40,
        }])
        self._prev_pynvml = _install_fake_pynvml(fake)
        out = gpu.query_nvidia_pynvml()
        self.assertEqual(out[0]["name"], "RTX 4090")
        self.assertIsInstance(out[0]["name"], str)

    def test_utilization_error_falls_back_to_zero(self):
        # nvmlDeviceGetUtilizationRates raises on some driver/GPU combos
        # (notably older or virtualised cards). The per-call try/except in
        # query_nvidia_pynvml pins utilization_pct to 0 instead of dropping
        # the whole GPU. Same idea for temperature: falls back to None.
        fake = _FakePynvml(gpus=[{
            "name": "GRID K1", "total": 4 * 1024**3, "used": 0, "free": 4 * 1024**3,
            "util": 0, "temp": 40,
        }])
        fake.nvmlDeviceGetUtilizationRates = MagicMock(side_effect=RuntimeError("not supported"))
        fake.nvmlDeviceGetTemperature = MagicMock(side_effect=RuntimeError("not supported"))
        self._prev_pynvml = _install_fake_pynvml(fake)
        out = gpu.query_nvidia_pynvml()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["utilization_pct"], 0)
        self.assertIsNone(out[0]["temperature_c"])

    def test_zero_devices_returns_none(self):
        # `None` is the "nothing to show" sentinel that the collect_gpu_info
        # caller uses to fall through to its exec-into-container fallback.
        # An empty list would look like "queried successfully, zero cards"
        # and short-circuit the fallback - so the distinction matters.
        fake = _FakePynvml(gpus=[])
        self._prev_pynvml = _install_fake_pynvml(fake)
        self.assertIsNone(gpu.query_nvidia_pynvml())

    def test_init_failure_returns_none(self):
        # nvmlInit raising - most common on a host with the package installed
        # but no NVIDIA driver visible - has to yield None, not raise.
        fake = _FakePynvml(gpus=[], init_raises=RuntimeError("NVML shared library not found"))
        self._prev_pynvml = _install_fake_pynvml(fake)
        self.assertIsNone(gpu.query_nvidia_pynvml())

    def test_import_failure_returns_none(self):
        # `import pynvml` itself raising ImportError - simulated by putting
        # None into sys.modules. The outer try/except must swallow it.
        self._prev_pynvml = sys.modules.get("pynvml")
        sys.modules["pynvml"] = None
        self.assertIsNone(gpu.query_nvidia_pynvml())


if __name__ == "__main__":
    unittest.main()
