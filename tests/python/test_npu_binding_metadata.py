"""Exercise the production Tensor/stream extraction helpers without NPU DMA."""
import unittest
import gc
import weakref
from types import SimpleNamespace

import _kvaf_binding_metadata_test as binding


class Device:
    def __init__(self, index=0, kind="npu"):
        self.index, self.kind = index, kind

    def __str__(self):
        return f"{self.kind}:{self.index}"


class Tensor:
    def __init__(self, shape=(2,), pointer=4096, elements=2, element_bytes=8,
                 dtype="torch.int64", device=None, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.device = device if device is not None else Device()
        self.pointer, self.elements, self.element_bytes = pointer, elements, element_bytes
        self.contiguous = contiguous

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self.elements

    def element_size(self):
        return self.element_bytes

    def is_contiguous(self):
        return self.contiguous


class NpuBindingMetadata(unittest.TestCase):
    def test_aicpu_workspace_resource_configuration(self):
        default = binding.ExecutionResources()
        self.assertEqual(default.gather_ddr_bytes, 0)  # Resolve the legacy layout after token/workers are known.
        self.assertEqual(default.scatter_hbm_bytes, 0)
        self.assertIsNone(default.gather_read_wave_wrs)
        resource = binding.ExecutionResources(gather_ddr_bytes=64 << 20, scatter_hbm_bytes=128 << 20,
                                              request_capacity=1000000)
        self.assertEqual(resource.gather_ddr_bytes, 64 << 20)
        self.assertEqual(resource.scatter_hbm_bytes, 128 << 20)
        self.assertEqual(resource.request_capacity, 1000000)
        for name in ("gather_ddr_bytes", "scatter_hbm_bytes"):
            self.assertEqual(getattr(binding.ExecutionResources(**{name: (1 << 64) - 1}), name), (1 << 64) - 1)
            for value in (-1, 1 << 64):
                with self.assertRaises((TypeError, OverflowError)):
                    binding.ExecutionResources(**{name: value})
        for limit in (1, 64, 96, 256, 4096):
            self.assertEqual(binding.ExecutionResources(gather_read_wave_wrs=limit).gather_read_wave_wrs, limit)
        for invalid in (-1, 1 << 32):
            with self.assertRaises((TypeError, OverflowError)):
                binding.ExecutionResources(gather_read_wave_wrs=invalid)

    def test_retained_registration_owns_original_storage_after_tensor_rebind(self):
        class Storage:
            pass

        tensor = Tensor()
        tensor.storage = Storage()
        tensor.untyped_storage = lambda: tensor.storage
        original = weakref.ref(tensor.storage)
        retained = binding.retain_storage(tensor)
        tensor.storage = Storage()  # Model Tensor.set_ replacing the active allocation.
        gc.collect()
        self.assertIsNotNone(original())
        self.assertIs(retained[0], tensor)
        del retained
        gc.collect()
        self.assertIsNone(original())

    def test_full_shape_and_raw_rank_accept_existing_layout_objects(self):
        tensor = Tensor(shape=(2, 3), elements=6)
        full = binding.view(tensor)
        raw = binding.view(tensor, keep_shape=False)
        self.assertEqual(full["shape"], [2, 3])
        self.assertEqual(raw["shape"], [])
        for view in (full, raw):
            self.assertEqual(view["dimensions"], 2)
            self.assertEqual(view["pointer"], 4096)
            self.assertEqual(view["bytes"], 48)
            self.assertIs(view["device"], tensor.device)
        self.assertEqual(binding.int64_view(Tensor())["bytes"], 16)

    def test_non_tuple_shapes_keep_the_previous_conversion_contract(self):
        for shape in ([2], (value for value in [2])):
            with self.subTest(shape=shape):
                self.assertEqual(binding.int64_view(Tensor(shape=shape))["dimensions"], 1)
        for shape in (("bad",), ["bad"]):
            with self.subTest(shape=shape):
                with self.assertRaises((RuntimeError, TypeError)):
                    binding.int64_view(Tensor(shape=shape))

        class Shape(tuple):
            def __iter__(self):
                return iter((1, 2))

        # Keep the same shape interpretation as the full-shape path, including
        # tuple subclasses used for Tensor metadata.
        shape = Shape((2,))
        self.assertEqual(binding.view(Tensor(shape=shape))["shape"], [1, 2])
        self.assertEqual(binding.view(Tensor(shape=shape), keep_shape=False)["dimensions"], 2)
        with self.assertRaises(ValueError):
            binding.int64_view(Tensor(shape=shape))

    def test_raw_metadata_validation(self):
        for tensor in (Tensor(shape=(1, 2)), Tensor(shape=()),
                       Tensor(dtype="torch.float64"), Tensor(element_bytes=4),
                       Tensor(device=Device(1)), Tensor(device=Device(kind="cpu")),
                       Tensor(pointer=4097), Tensor(pointer=0), Tensor(elements=0),
                       Tensor(elements=1 << 63), Tensor(contiguous=False)):
            with self.subTest(tensor=vars(tensor)):
                with self.assertRaises(ValueError):
                    binding.int64_view(tensor)
        for tensor in (None, [0, 8], SimpleNamespace(data_ptr=lambda: 4096)):
            with self.assertRaisesRegex(TypeError, "metadata"):
                binding.int64_view(tensor)
        tensor = Tensor()
        del tensor.dtype
        with self.assertRaisesRegex(ValueError, "torch.int64"):
            binding.int64_view(tensor)

    def test_layout_is_rechecked_when_the_same_object_changes(self):
        tensor = Tensor()
        self.assertEqual(binding.int64_view(tensor)["pointer"], 4096)
        tensor.pointer = 8192
        self.assertEqual(binding.int64_view(tensor)["pointer"], 8192)
        tensor.dtype = "torch.float64"
        with self.assertRaises(ValueError):
            binding.int64_view(tensor)
        tensor.dtype = "torch.int64"
        tensor.device.index = 1
        with self.assertRaises(ValueError):
            binding.int64_view(tensor)

    def test_device_is_observed_once_per_extraction(self):
        class ObservedTensor(Tensor):
            reads = 0

            @property
            def device(self):
                self.reads += 1
                return Device()

            @device.setter
            def device(self, value):
                pass

        tensor = ObservedTensor()
        binding.int64_view(tensor)
        self.assertEqual(tensor.reads, 1)

    def test_stream_handle_validation(self):
        self.assertEqual(binding.stream(SimpleNamespace(npu_stream=4096)), 4096)
        with self.assertRaisesRegex(TypeError, "Stream"):
            binding.stream(object())
        with self.assertRaisesRegex(ValueError, "null"):
            binding.stream(SimpleNamespace(npu_stream=0))

    def test_metadata_order_probe_checks_every_array(self):
        for first in range(3):
            times = binding.profile_arrays(Tensor(), Tensor(), Tensor(), first=first)
            self.assertEqual(len(times), 3)
            self.assertTrue(all(value >= 0 for value in times))
            with self.assertRaises(ValueError):
                binding.profile_arrays(Tensor(), Tensor(dtype="torch.float64"), Tensor(), first=first)


if __name__ == "__main__":
    unittest.main()
