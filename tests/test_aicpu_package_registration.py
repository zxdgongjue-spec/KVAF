import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_HELPER = ROOT / "scripts" / "a5_aicpu_package_registration.sh"
PACKAGE_NAME = "cann-kvaf-direct-urma-compat.tar.gz"
PACKAGE_PATH = "opp/vendors/cust/op_impl/aicpu/kernel"

TARGET_BLOCK = """\
name:cann-kvaf-direct-urma-compat.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/op_impl/aicpu/kernel
load_as_per_soc:false
"""

OTHER_STANZAS = """\
# Keep comments and unrelated package registrations unchanged.
name:cann-hybm-compat.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/op_impl/aicpu/kernel
load_as_per_soc:false

name:cann-hcomm-compat.tar.gz
install_path:1
optional:false
package_path:compat
load_as_per_soc:true
"""


class AicpuPackageRegistrationTests(unittest.TestCase):
    def run_helper(
        self,
        function: str,
        registry: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'set -Eeuo pipefail\nsource "$1"\n"$2" "$3" "$4" "$5"',
                "kvaf-aicpu-registration-test",
                str(REGISTRATION_HELPER),
                function,
                str(registry),
                PACKAGE_NAME,
                PACKAGE_PATH,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_helper_succeeds(
        self,
        function: str,
        registry: Path,
    ) -> None:
        result = self.run_helper(function, registry)
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def assert_helper_fails(
        self,
        function: str,
        registry: Path,
    ) -> None:
        result = self.run_helper(function, registry)
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_first_append_preserves_all_other_stanzas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"
            registry.write_text(OTHER_STANZAS, encoding="utf-8")

            self.assert_helper_succeeds(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertEqual(
                registry.read_text(encoding="utf-8"),
                OTHER_STANZAS + TARGET_BLOCK,
            )
            self.assert_helper_succeeds(
                "kvaf_aicpu_registration_is_exact", registry
            )

    def test_first_append_to_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"
            registry.touch()

            self.assert_helper_succeeds(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertEqual(
                registry.read_text(encoding="utf-8"), TARGET_BLOCK
            )

    def test_install_rejects_missing_registry_without_creating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"

            self.assert_helper_fails(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertFalse(registry.exists())

    def test_install_rejects_missing_registry_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_parent = Path(temporary_dir) / "previously-missing-conf"
            registry = missing_parent / "ascend_package_load.ini"

            self.assert_helper_fails(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertFalse(missing_parent.exists())
            self.assertFalse(registry.exists())

    def test_install_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"
            registry.write_text(OTHER_STANZAS, encoding="utf-8")

            self.assert_helper_succeeds(
                "kvaf_aicpu_install_registration", registry
            )
            first_content = registry.read_bytes()
            first_mode = stat.S_IMODE(registry.stat().st_mode)

            self.assert_helper_succeeds(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertEqual(registry.read_bytes(), first_content)
            self.assertEqual(
                stat.S_IMODE(registry.stat().st_mode), first_mode
            )
            self.assertEqual(
                registry.read_text(encoding="utf-8").count(
                    f"name:{PACKAGE_NAME}\n"
                ),
                1,
            )

    def test_exact_check_rejects_duplicate_target_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"
            registry.write_text(
                OTHER_STANZAS + TARGET_BLOCK + TARGET_BLOCK,
                encoding="utf-8",
            )

            self.assert_helper_fails(
                "kvaf_aicpu_registration_is_exact", registry
            )

    def test_exact_check_rejects_malformed_target_blocks(self) -> None:
        malformed_blocks = {
            "missing_line": TARGET_BLOCK.replace("optional:true\n", ""),
            "wrong_value": TARGET_BLOCK.replace(
                "optional:true", "optional:false"
            ),
            "wrong_order": TARGET_BLOCK.replace(
                "install_path:2\noptional:true\n",
                "optional:true\ninstall_path:2\n",
            ),
            "extra_line": TARGET_BLOCK + "unexpected:true\n",
        }
        for label, malformed_block in malformed_blocks.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    registry = (
                        Path(temporary_dir) / "ascend_package_load.ini"
                    )
                    registry.write_text(
                        OTHER_STANZAS + malformed_block,
                        encoding="utf-8",
                    )

                    self.assert_helper_fails(
                        "kvaf_aicpu_registration_is_exact", registry
                    )

    def test_install_rejects_ambiguous_target_without_mutation(self) -> None:
        unsafe_registries = {
            "duplicate": OTHER_STANZAS + TARGET_BLOCK + TARGET_BLOCK,
            "malformed": OTHER_STANZAS
            + TARGET_BLOCK.replace("optional:true", "optional:false"),
        }
        for label, initial_content in unsafe_registries.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    registry = (
                        Path(temporary_dir) / "ascend_package_load.ini"
                    )
                    registry.write_text(initial_content, encoding="utf-8")
                    initial_bytes = registry.read_bytes()

                    self.assert_helper_fails(
                        "kvaf_aicpu_install_registration", registry
                    )
                    self.assertEqual(registry.read_bytes(), initial_bytes)

    def test_exact_check_rejects_missing_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"

            self.assert_helper_fails(
                "kvaf_aicpu_registration_is_exact", registry
            )

    def test_exact_check_and_install_reject_registry_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            target = root / "real-ascend-package-load.ini"
            target.write_text(OTHER_STANZAS, encoding="utf-8")
            target_before = target.read_bytes()
            registry = root / "ascend_package_load.ini"
            registry.symlink_to(target)

            self.assert_helper_fails(
                "kvaf_aicpu_registration_is_exact", registry
            )
            self.assert_helper_fails(
                "kvaf_aicpu_install_registration", registry
            )
            self.assertTrue(registry.is_symlink())
            self.assertEqual(target.read_bytes(), target_before)

    def test_install_preserves_existing_registry_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            registry = Path(temporary_dir) / "ascend_package_load.ini"
            registry.write_text(OTHER_STANZAS, encoding="utf-8")
            os.chmod(registry, 0o640)

            self.assert_helper_succeeds(
                "kvaf_aicpu_install_registration", registry
            )

            self.assertEqual(
                stat.S_IMODE(registry.stat().st_mode), 0o640
            )


if __name__ == "__main__":
    unittest.main()
