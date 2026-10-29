import contextlib
import io
import unittest

from test_aiv_sequential_batch_get import shutdown_runtime


class Runtime:
    initialized = True

    def shutdown(self):
        raise RuntimeError("cleanup failed")


class CleanupTests(unittest.TestCase):
    def test_original_exception_survives_cleanup_failure(self):
        original = RuntimeError("execution failed")
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            with self.assertRaises(RuntimeError) as raised:
                try:
                    raise original
                finally:
                    shutdown_runtime(Runtime())
        self.assertIs(raised.exception, original)
        self.assertIn("shutdown also failed", log.getvalue())
        self.assertIn("cleanup failed", log.getvalue())

    def test_cleanup_failure_without_original_still_raises(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            shutdown_runtime(Runtime())

    def test_uninitialized_runtime_does_not_shutdown(self):
        runtime = Runtime()
        runtime.initialized = False
        shutdown_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
