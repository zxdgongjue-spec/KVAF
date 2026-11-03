"""Binding contracts without NPU hardware: identities, serialization and array types."""
import pickle
import unittest

import numpy as np
import kvaf

try:
    import torch
except ImportError:
    torch = None


def restored(cls, state):
    # Exercise the same reconstruction entrypoint used by Connector pickle IPC.
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


class NpuTensorLayout:
    """Host storage with NPU metadata for layout checks only; never executes DMA."""
    class Device:
        def __init__(self, index):
            self.index = index

        def __str__(self):
            return f"npu:{self.index}"

    def __init__(self, shape=(2,), dtype="torch.int64", contiguous=True, device=0):
        self.storage = np.zeros(shape, dtype=np.int64)
        self.shape = shape
        self.dtype = dtype
        self.device = self.Device(device)
        self.contiguous = contiguous

    def data_ptr(self):
        return self.storage.ctypes.data

    def numel(self):
        return self.storage.size

    def element_size(self):
        return 8

    def is_contiguous(self):
        return self.contiguous


class ConnectorBindings(unittest.TestCase):
    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_base_and_ddr_initialization_are_independent(self):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        self.assertFalse(runtime.initialized)
        self.assertFalse(runtime.ddr_initialized)
        with self.assertRaisesRegex(RuntimeError, "call initialize before initialize_ddr"):
            runtime.initialize_ddr(4096, 4096)
        self.assertIsNone(runtime.initialize())
        self.assertIsNone(runtime.initialize())
        self.assertTrue(runtime.initialized)
        self.assertFalse(runtime.ddr_initialized)
        with self.assertRaisesRegex(ValueError, "link_to before initialize_ddr"):
            runtime.initialize(ddr_bytes=4096, slot_size=4096)
        with self.assertRaisesRegex(RuntimeError, "initialize_ddr before PreparePut"):
            runtime.prepare_put([0], [8])
        with self.assertRaisesRegex(RuntimeError, "DDR is not initialized"):
            _ = runtime.remote_region
        with self.assertRaisesRegex(RuntimeError, "DDR is not initialized"):
            runtime.batch_put([0], None, [0], [8])
        self.assertIsNone(runtime.shutdown_ddr())
        self.assertTrue(runtime.initialized)
        self.assertFalse(runtime.ddr_initialized)
        self.assertIsNone(runtime.shutdown())
        self.assertFalse(runtime.initialized)
        self.assertIsNone(runtime.shutdown())

    def test_descriptors_and_ready_survive_connector_ipc(self):
        cpu = restored(kvaf.CpuPullSourceDesc, (12, 4096, 2, 99))
        npu = restored(kvaf.NpuPullSourceDesc, (13, 8192, 88))
        for source in (cpu, npu):
            ready = restored(kvaf.ReadyToken, (source, 7))
            copy = pickle.loads(pickle.dumps(ready))
            self.assertEqual(copy.sequence, 7)
            self.assertEqual(type(copy.source), type(source))
            self.assertEqual(copy.source.source_id, source.source_id)
            self.assertEqual(copy.source.source_bytes, source.source_bytes)
            self.assertFalse(hasattr(copy, "readable_ranges"))
            with self.assertRaises(AttributeError):
                copy.sequence = 8
        self.assertEqual(pickle.loads(pickle.dumps(cpu)).region_epoch, 99)
        self.assertEqual(pickle.loads(pickle.dumps(npu)).source_epoch, 88)

    def test_cpu_multi_mr_identity_survives_connector_ipc(self):
        sizes = [4096 + i * 64 for i in range(78)]
        cpu = restored(kvaf.CpuPullSourceDesc, (12, sizes[0], 2, 99, sizes))
        ready = restored(kvaf.ReadyToken, (cpu, 7, 78))
        copy = pickle.loads(pickle.dumps(ready))
        self.assertEqual(copy.mr_id, 78)
        self.assertEqual(copy.source.mr_count, 78)
        self.assertEqual(copy.source.mr_bytes, sizes)
        # Returned lists cannot mutate the identity that is stored in KVAF.
        copy.source.mr_bytes.clear()
        self.assertEqual(copy.source.mr_count, 78)
        with self.assertRaises(AttributeError):
            copy.mr_id = 1
        with self.assertRaises(AttributeError):
            copy.source.mr_bytes = [1]
        single = restored(kvaf.CpuPullSourceDesc, (12, 4096, 2, 99))
        self.assertEqual(single.mr_count, 1)
        self.assertEqual(restored(kvaf.ReadyToken, (single, 7)).mr_id, 1)

    def test_npu_multi_mr_identity_survives_connector_ipc(self):
        sizes = [4096 + i * 64 for i in range(78)]
        source = restored(kvaf.NpuPullSourceDesc, (13, sizes[0], 88, sizes))
        ready = restored(kvaf.ReadyToken, (source, 9, 78))
        copy = pickle.loads(pickle.dumps(ready))
        self.assertEqual(copy.mr_id, 78)
        self.assertEqual(copy.source.mr_count, 78)
        self.assertEqual(copy.source.mr_bytes, sizes)
        copy.source.mr_bytes.clear()
        self.assertEqual(copy.source.mr_count, 78)
        with self.assertRaises(AttributeError):
            copy.mr_id = 1
        self.assertEqual(restored(kvaf.NpuPullSourceDesc, (13, 8192, 88)).mr_count, 1)

    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_only_current_pull_interfaces_are_exported(self):
        for name in ("HbmSource", "HbmReadSource", "OffloadRuntime", "CpuPullSourceHandle", "CpuPullTask"):
            self.assertFalse(hasattr(kvaf, name))
        self.assertFalse(hasattr(kvaf.PutHandle, "task"))
        self.assertTrue(hasattr(kvaf.PutHandle, "submitted"))
        for name in ("offload", "offload_from", "resolve_offload", "release_offload_source"):
            self.assertFalse(hasattr(kvaf.NpuRuntime, name))
        self.assertTrue(hasattr(kvaf.NpuRuntime, "batch_cpu_pull"))
        self.assertTrue(hasattr(kvaf.NpuRuntime, "resolve_cpu_pull"))
        self.assertTrue(hasattr(kvaf.NpuRuntime, "listen_hbm"))
        self.assertFalse(hasattr(kvaf.NpuRuntime, "attach_hbm"))
        self.assertFalse(hasattr(kvaf.NpuRuntime, "detach_hbm"))
        self.assertFalse(hasattr(kvaf.NpuPullHbmSource, "listen"))
        with self.assertRaises(TypeError):
            kvaf.NpuPullHbmSource(0).register(None)  # receiver is mandatory
        for cls in (kvaf.CpuPullHbmSource, kvaf.NpuPullHbmSource):
            source = cls(0)
            with self.assertRaises(TypeError):
                source.ready([0], [8])
            with self.assertRaises(TypeError):
                source.release_batch()
            with self.assertRaises(TypeError):
                source.cancel_batch()
        with self.assertRaises(TypeError):
            kvaf.NpuRuntime(enable_cpu_offload=True).batch_get_raw_addr(1, {}, None, [0])

    def call_with_arrays(self, arrays):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        source = restored(kvaf.NpuPullSourceDesc, (13, 8192, 88))
        ready = restored(kvaf.ReadyToken, (source, 7))
        runtime.batch_get_raw_addr(ready, *arrays, output=None)

    @unittest.skipUnless(hasattr(kvaf, "NpuRuntime"), "NPU Runtime binding not built")
    def test_recovery_and_cancel_accept_connector_identity_objects(self):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        source = restored(kvaf.NpuPullSourceDesc, (13, 8192, 88))
        ready = restored(kvaf.ReadyToken, (source, 7))
        # Typed arguments reach Runtime without an application-local link ID.
        with self.assertRaisesRegex(RuntimeError, "no HBM reader"):
            runtime.resolve_raw_addr(ready=ready)
        with self.assertRaisesRegex(RuntimeError, "not initialized|requires AIV"):
            runtime.cancel_raw_addr(ready=ready)
        for method in (runtime.resolve_raw_addr, runtime.cancel_raw_addr):
            with self.assertRaises(TypeError):
                method(1)

    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_cpu_pull_lengths_remain_batch_put_compatible(self):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        ready = restored(kvaf.ReadyToken, (restored(kvaf.CpuPullSourceDesc, (12, 4096, 2, 99)), 1))
        accepted = [[8, 8], (8, 8), np.array([8, 8], dtype=np.uint64),
                    np.array([8, 8], dtype=np.int64), np.arange(4, dtype=np.uint64)[::2] + 1]
        if torch is not None:
            accepted.append(torch.tensor([8, 8], dtype=torch.int64, device="cpu"))
        for value in accepted:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(RuntimeError, "initialize_ddr before PreparePut"):
                    runtime.prepare_put([1, 2], value)
                with self.assertRaisesRegex(RuntimeError, "initialize_ddr before batch_cpu_pull"):
                    runtime.batch_cpu_pull([1, 2], ready, [0, 64], value, timeout=0)

    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_batch_cpu_pull_requires_ready_instead_of_local_tensor(self):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        ready = restored(kvaf.ReadyToken, (restored(kvaf.CpuPullSourceDesc, (12, 4096, 2, 99)), 1))
        with self.assertRaises(TypeError):
            runtime.batch_cpu_pull([0], NpuTensorLayout(), [0], [8])
        with self.assertRaises(TypeError):
            runtime.batch_cpu_pull([0], ready, [0], [8], stream=None)
        with self.assertRaisesRegex(RuntimeError, "initialize_ddr before batch_cpu_pull"):
            runtime.batch_cpu_pull([0], ready, [0], [8], profile=True)
        with self.assertRaisesRegex(RuntimeError, "initialize_ddr before resolve_cpu_pull"):
            runtime.resolve_cpu_pull(timeout=0)

    def test_raw_arrays_require_npu_layout(self):
        valid = NpuTensorLayout()
        rejected = [[0, 8], np.array([0, 8], dtype=np.uint64),
                    NpuTensorLayout(dtype="torch.float64"), NpuTensorLayout(shape=(1, 2)),
                    NpuTensorLayout(contiguous=False), NpuTensorLayout(device=1),
                    NpuTensorLayout(shape=(0,))]
        if torch is not None:
            rejected.append(torch.tensor([0, 8], dtype=torch.int64, device="cpu"))
        for position, name in enumerate(("src_offsets", "dst_offsets", "lengths")):
            for value in rejected:
                arrays = [valid, valid, valid]
                arrays[position] = value
                with self.subTest(array=name, value=repr(value)):
                    with self.assertRaisesRegex((ValueError, TypeError), name):
                        self.call_with_arrays(arrays)
        with self.assertRaisesRegex(TypeError, "output"):
            self.call_with_arrays([valid, valid, valid])

    def test_raw_arrays_require_equal_bounded_counts(self):
        for arrays in ((NpuTensorLayout(), NpuTensorLayout(shape=(1,)), NpuTensorLayout()),
                       (NpuTensorLayout(shape=(4097,)),) * 3):
            with self.assertRaisesRegex(ValueError, "equal lengths"):
                self.call_with_arrays(arrays)


if __name__ == "__main__":
    unittest.main()
