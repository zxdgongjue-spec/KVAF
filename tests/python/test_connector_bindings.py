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

    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_only_current_pull_interfaces_are_exported(self):
        for name in ("HbmSource", "HbmReadSource", "OffloadRuntime"):
            self.assertFalse(hasattr(kvaf, name))
        for name in ("offload", "offload_from", "resolve_offload", "release_offload_source"):
            self.assertFalse(hasattr(kvaf.NpuRuntime, name))
        self.assertTrue(hasattr(kvaf.NpuRuntime, "batch_cpu_pull"))
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
        runtime.batch_get_raw_addr(1, ready, *arrays, output=None)

    @unittest.skipUnless(hasattr(kvaf, "CpuPullHbmSource"), "CPU Offload binding not built")
    def test_cpu_pull_lengths_remain_batch_put_compatible(self):
        runtime = kvaf.NpuRuntime(enable_cpu_offload=True)
        accepted = [[8, 8], (8, 8), np.array([8, 8], dtype=np.uint64),
                    np.array([8, 8], dtype=np.int64), np.arange(4, dtype=np.uint64)[::2] + 1]
        if torch is not None:
            accepted.append(torch.tensor([8, 8], dtype=torch.int64, device="cpu"))
        for value in accepted:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(RuntimeError, "initialize Runtime before PreparePut"):
                    runtime.prepare_put([1, 2], value)

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
