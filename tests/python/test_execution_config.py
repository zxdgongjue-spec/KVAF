"""Portable surface checks: no device or torch dependency."""
import unittest
import kvaf


class ExecutionConfigTests(unittest.TestCase):
    def test_complete_values(self):
        put = kvaf.PutExecutionConfig(transport="urma", executor="aicpu")
        get = kvaf.GetExecutionConfig(transport="ubmem", executor="aiv", algorithm="direct")
        self.assertEqual((put.transport, put.executor), ("urma", "aicpu"))
        self.assertEqual((get.transport, get.executor, get.algorithm), ("ubmem", "aiv", "direct"))
        with self.assertRaises(AttributeError):
            get.algorithm = "scatter"

    def test_invalid_strings(self):
        for value in ("auto", "", "URMA", "urma+ubmem"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                kvaf.PutExecutionConfig(value, "aiv")
        with self.assertRaises(ValueError):
            kvaf.PutExecutionConfig("urma", "cpu")
        with self.assertRaises(ValueError):
            kvaf.GetExecutionConfig("urma", "aiv", "auto")

    def test_reserved_algorithm_has_a_value(self):
        # Availability is checked against the runtime's enabled provider set.
        get = kvaf.GetExecutionConfig("urma", "aiv", "scatter")
        self.assertEqual(get.algorithm, "scatter")


if __name__ == "__main__":
    unittest.main()
