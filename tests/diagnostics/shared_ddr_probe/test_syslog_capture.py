"""Check logging interception without UMDK; Linux also exercises LD_PRELOAD."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SyslogCaptureTest(unittest.TestCase):
    def test_entry_points_and_errno(self):
        compiler = shutil.which("c++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = Path(__file__).with_name("syslog_capture.cc")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.cc"
            fixture.write_text(r'''
#include <cerrno>
#include <cstdarg>
#include <cassert>
#include <cstring>
extern "C" void syslog(int, const char*, ...);
extern "C" void vsyslog(int, const char*, va_list);
extern "C" void __syslog_chk(int, int, const char*, ...);
extern "C" void __vsyslog_chk(int, int, const char*, va_list);
void variadic(bool checked, const char* format, ...) {
  va_list args; va_start(args, format);
  if (checked) __vsyslog_chk(3, 1, format, args);
  else vsyslog(3, format, args);
  va_end(args);
}
int main() {
  errno = EACCES;
  syslog(3, "normal value=%d", 7); assert(errno == EACCES);
  variadic(false, "variadic value=%d", 8); assert(errno == EACCES);
  __syslog_chk(3, 1, "checked value=%d", 9); assert(errno == EACCES);
  variadic(true, "checked_variadic value=%d", 10); assert(errno == EACCES);
  syslog(3, "saved_error=%m"); assert(errno == EACCES);
  char long_line[8000]; std::memset(long_line, 'X', sizeof(long_line)-1);
  long_line[sizeof(long_line)-1] = 0;
  syslog(3, "%s", long_line); assert(errno == EACCES);
  return 0;
}
''')
            binary = root / "fixture"
            env = dict(os.environ, LC_ALL="C")
            if sys.platform.startswith("linux"):
                library = root / "capture.so"
                subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(source),
                                "-o", str(library)], check=True, capture_output=True)
                subprocess.run([compiler, str(fixture), "-o", str(binary)],
                               check=True, capture_output=True)
                env["LD_PRELOAD"] = str(library)
            else:
                # Local macOS verifies the same entry points by direct linkage;
                # it does not certify Linux dynamic-loader interposition.
                subprocess.run([compiler, "-std=c++17", str(fixture), str(source),
                                "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], env=env, capture_output=True,
                                    text=True, check=True, timeout=30)
            self.assertIn("PROBE_SYSLOG_CAPTURE_READY", result.stderr)
            for message in ("normal value=7", "variadic value=8", "checked value=9",
                            "checked_variadic value=10", "saved_error="):
                self.assertIn(message, result.stderr)
            if sys.platform.startswith("linux"):
                # %m is a glibc printf extension; macOS vsnprintf lacks it.
                self.assertIn("saved_error=Permission denied", result.stderr)
            self.assertIn("[truncated]", result.stderr)
            self.assertTrue(all(len(line.encode()) <= 4096
                                for line in result.stderr.splitlines(keepends=True)))


if __name__ == "__main__":
    unittest.main()
