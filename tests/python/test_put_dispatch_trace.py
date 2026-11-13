"""Portable checks for nested Put stages and independent trace populations."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from summarize_put_dispatch_trace import summarize


class PutDispatchTraceTests(unittest.TestCase):
    def test_last_samples_keep_backends_paths_and_recovery_separate(self):
        base = dict(transport="urma", executor="aiv", path="input", event="batch_put", result="0",
                    total_ns="50000", provider_ns="40000", provider_prepare_ns="10000",
                    provider_registration_ns="8000", provider_metadata_ns="2000",
                    provider_submit_ns="5000", provider_completion_ns="20000", provider_cleanup_ns="1000")
        rows = [dict(base, total_ns="999000"), base,
                dict(base, executor="aicpu", path="ordinary"),
                dict(base, path="legacy"),
                dict(base, event="recover", result="4")]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize(rows, last=1)
        text = output.getvalue()
        self.assertEqual(text.count("PUT_DISPATCH_TIMING"), 4)
        self.assertNotIn("999.00", text)
        self.assertIn("executor=aiv path=input event=batch_put result=0 samples=1", text)
        self.assertIn("executor=aiv path=input event=recover result=4 samples=1", text)
        self.assertIn("executor=aicpu path=ordinary", text)
        self.assertIn("provider_other: p50=2.00", text)
        self.assertIn("provider_registration: p50=8.00", text)
        self.assertIn("outside_provider: p50=10.00", text)

    def test_invalid_or_empty_trace_is_not_reported_as_success(self):
        script = Path(__file__).with_name("summarize_put_dispatch_trace.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            for contents in ("event,total_ns\nsubmit,100\n",
                    "transport,executor,path,event,result,total_ns,provider_ns\n"):
                path.write_text(contents)
                result = subprocess.run([sys.executable, str(script), str(path)],
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("PUT_DISPATCH_TIMING", result.stdout)


if __name__ == "__main__":
    unittest.main()
