"""Call the production constructor registration using a device-free probe."""
import unittest

from _kvaf_constructor_test import NpuRuntime


class ConstructorTests(unittest.TestCase):
    def test_original_defaults(self):
        self.assertEqual(NpuRuntime("get.o", 656, 128).values,
                         ("get.o", 656, 128, 0, 2, "", 2, [], False, 2048, "", [], 2048, "urma", ""))

    def test_original_six_positional_arguments(self):
        self.assertEqual(NpuRuntime("get.o", 656, 128, 7, 5, "put.o").values[:6],
                         ("get.o", 656, 128, 7, 5, "put.o"))

    def test_all_original_positional_arguments_including_transport(self):
        old = ("get.o", 1024, 64, 3, 4, "put.o", 6, ["eid"], True, 4096,
               "udma0", ["udma0", "udma1"], 8192, "ubmem")
        self.assertEqual(NpuRuntime(*old).values, old + ("",))

    def test_keywords_and_new_mode(self):
        for mode in ("gather_read", "cpu_push_scatter"):
            with self.subTest(mode=mode):
                runtime = NpuRuntime(batch_get_kernel_object_path="", batch_get_token_bytes=656,
                                     batch_get_tokens_per_block=2, device_id=7, npu_core_count=5,
                                     transport="urma", aicpu_batch_get_mode=mode)
                self.assertEqual(runtime.values[:5], ("", 656, 2, 7, 5))
                self.assertEqual(runtime.values[-2:], ("urma", mode))
                self.assertEqual(NpuRuntime("", 656, 2, 7, 5, aicpu_batch_get_mode=mode).values,
                                 runtime.values)

    def test_new_mode_is_keyword_only(self):
        with self.assertRaises(TypeError):
            NpuRuntime("", 656, 2, 7, 5, "", 2, [], False, 2048, "", [], 2048,
                       "urma", "gather_read")


if __name__ == "__main__":
    unittest.main()
