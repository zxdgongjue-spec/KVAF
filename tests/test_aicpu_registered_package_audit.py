from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_a5_registered_aicpu_packages.sh"


FAKE_READELF = r"""#!/bin/sh
mode=$1
for argument do
  target=$argument
done
marker=$(/usr/bin/head -c 1 "$target")
case "$mode" in
  -h)
    [ "$marker" != X ] || exit 1
    echo "ELF Header:"
    ;;
  -d)
    echo "Dynamic section at offset 0x100 contains 2 entries:"
    case "$marker" in
      A|B)
        soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so
        ;;
      C)
        soname=liburma.so.0
        ;;
      D)
        soname=libummu.so.1
        ;;
      E)
        soname=libudma.so.2
        ;;
      *)
        soname=libunknown.so
        ;;
    esac
    printf ' 0x000000000000000e (SONAME) Library soname: [%s]\n' "$soname"
    printf ' 0x0000000000000001 (NEEDED) Shared library: [libdep_%s.so]\n' "$marker"
    ;;
  -n)
    printf '    Build ID: %s%s%s%s\n' "$marker" "$marker" "$marker" "$marker"
    ;;
  --dyn-syms)
    cat <<EOF
Symbol table '.dynsym' contains 3 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000001000     8 FUNC    GLOBAL DEFAULT    7 export_${marker}
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND import_${marker}
     3: 0000000000002000     8 FUNC    WEAK   PROTECTED ABS weak_${marker}
EOF
    ;;
  *)
    exit 2
    ;;
esac
"""


class RegisteredAicpuPackageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "ascend"
        self.root.mkdir()
        self.registry = self.root / "conf" / "ascend_package_load.ini"
        self.registry.parent.mkdir()
        self.tools = Path(self.temporary.name) / "tools"
        self.tools.mkdir()
        readelf = self.tools / "readelf"
        readelf.write_text(FAKE_READELF, encoding="utf-8")
        readelf.chmod(0o755)

    def write_archive(
        self,
        package_path: str,
        package_name: str,
        members: dict[str, bytes],
    ) -> Path:
        archive_path = self.root / package_path / package_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            for member, content in members.items():
                info = tarfile.TarInfo(member)
                info.mode = 0o644
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return archive_path

    def write_plain_tar_with_gzip_suffix(
        self,
        package_path: str,
        package_name: str,
        members: dict[str, bytes],
    ) -> Path:
        archive_path = self.root / package_path / package_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w") as archive:
            for member, content in members.items():
                info = tarfile.TarInfo(member)
                info.mode = 0o644
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return archive_path

    def write_archive_with_symlink(
        self,
        package_path: str,
        package_name: str,
        target: str,
        alias: str,
        content: bytes,
    ) -> Path:
        archive_path = self.root / package_path / package_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            target_info = tarfile.TarInfo(target)
            target_info.mode = 0o644
            target_info.size = len(content)
            archive.addfile(target_info, io.BytesIO(content))
            alias_info = tarfile.TarInfo(alias)
            alias_info.type = tarfile.SYMTYPE
            alias_info.linkname = Path(target).name
            archive.addfile(alias_info)
        return archive_path

    def write_archive_with_symlink_chain(
        self,
        package_path: str,
        package_name: str,
        target: str,
        aliases: list[tuple[str, str]],
        content: bytes,
    ) -> Path:
        archive_path = self.root / package_path / package_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            target_info = tarfile.TarInfo(target)
            target_info.mode = 0o644
            target_info.size = len(content)
            archive.addfile(target_info, io.BytesIO(content))
            for alias, linkname in aliases:
                alias_info = tarfile.TarInfo(alias)
                alias_info.type = tarfile.SYMTYPE
                alias_info.linkname = linkname
                archive.addfile(alias_info)
        return archive_path

    def run_audit(self, *extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [str(self.tools), env.get("PATH", os.defpath)]
        )
        env["ASCEND_HOME_PATH"] = str(self.root)
        return subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                str(AUDIT),
                "--registry",
                str(self.registry),
                *extra,
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_success(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_script_has_valid_bash_syntax_and_no_runtime_loader(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(AUDIT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = AUDIT.read_text(encoding="utf-8")
        for forbidden in (
            "ldd ",
            "npu-smi",
            "ascend-dmi",
            "aclInit",
            "aclrt",
            "rtLaunch",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("readelf -d", source)
        self.assertIn("readelf --dyn-syms --wide", source)

    def test_comments_whitespace_and_all_library_families_are_audited(
        self,
    ) -> None:
        first = self.write_archive(
            "packages/one",
            "first.tar.gz",
            {
                "aicpu_kernels_device/libkvaf_rt_probe.so": b"A-first",
                "runtime/liburma.so.0": b"C-urma",
                "runtime/libkvaf_manifest.txt": b"ignored",
            },
        )
        second = self.write_archive(
            "packages/two",
            "second.tar.gz",
            {
                "lib/libummu_vendor.so.1": b"D-ummu",
                "lib/libudma.so.2": b"E-udma",
            },
        )
        self.registry.write_text(
            textwrap.dedent(
                """
                # A full-line comment and flexible whitespace are accepted.
                  name : first.tar.gz    ; inline comment
                  install_path : 2
                  package_path : packages/one   # first directory

                ; another full-line comment
                name:second.tar.gz
                package_path:packages/two
                # package_path:packages/not-effective
                """
            ),
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(f"PACKAGE\tarchive_path={first}", result.stdout)
        self.assertIn(f"PACKAGE\tarchive_path={second}", result.stdout)
        member_sha = hashlib.sha256(b"A-first").hexdigest()
        self.assertIn(
            "\tmember=aicpu_kernels_device/libkvaf_rt_probe.so"
            f"\tsha256={member_sha}\tgnu_build_id=aaaa",
            result.stdout,
        )
        self.assertIn("\tdt_soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so", result.stdout)
        self.assertIn(
            "DT_NEEDED\t"
            f"archive_path={first}\t"
            "member=aicpu_kernels_device/libkvaf_rt_probe.so\t"
            "value=libdep_A.so",
            result.stdout,
        )
        self.assertIn(
            "DYNAMIC_EXPORT\t"
            f"archive_path={first}\t"
            "member=aicpu_kernels_device/libkvaf_rt_probe.so\t"
            "value=export_A",
            result.stdout,
        )
        self.assertIn("value=weak_A", result.stdout)
        self.assertNotIn("value=import_A", result.stdout)
        for member in (
            "runtime/liburma.so.0",
            "lib/libummu_vendor.so.1",
            "lib/libudma.so.2",
        ):
            self.assertIn(f"\tmember={member}\t", result.stdout)
        self.assertIn(
            "SUMMARY\tregistered_packages=2\tlibraries=4\t"
            "conflicts=0\tcandidate_conflicts=0\terrors=0",
            result.stdout,
        )
        self.assertEqual(result.stderr, "")

    def test_same_runtime_soname_with_different_sha_is_a_conflict(self) -> None:
        first = self.write_archive(
            "packages",
            "one.tar.gz",
            {"a/libkvaf_rt_aaaaaaaaaaaaaaaa.so": b"A-generation-one"},
        )
        second = self.write_archive(
            "packages",
            "two.tar.gz",
            {"b/libkvaf_rt_aaaaaaaaaaaaaaaa.so": b"B-generation-two"},
        )
        self.registry.write_text(
            textwrap.dedent(
                """
                name:one.tar.gz
                package_path:packages
                name:two.tar.gz
                package_path:packages
                """
            ),
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "CONFLICT\truntime_soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so\t"
            "distinct_sha256=2",
            result.stderr,
        )
        for archive, member, content in (
            (first, "a/libkvaf_rt_aaaaaaaaaaaaaaaa.so", b"A-generation-one"),
            (second, "b/libkvaf_rt_aaaaaaaaaaaaaaaa.so", b"B-generation-two"),
        ):
            self.assertIn(f"\tarchive_path={archive}\tmember={member}", result.stderr)
            self.assertIn(hashlib.sha256(content).hexdigest(), result.stderr)
        self.assertIn(
            "\tconflicts=1\tcandidate_conflicts=0\terrors=0",
            result.stdout,
        )

    def test_versioned_library_symlink_is_reported_and_target_is_audited(
        self,
    ) -> None:
        archive = self.write_archive_with_symlink(
            "packages",
            "official.tar.gz",
            "runtime/liburma.so.0",
            "runtime/liburma.so",
            b"C-official",
        )
        self.registry.write_text(
            "name:official.tar.gz\npackage_path:packages\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            f"LIBRARY_ALIAS\tarchive_path={archive}\t"
            "member=runtime/liburma.so\ttype=l\t"
            "target=runtime/liburma.so.0",
            result.stdout,
        )
        self.assertIn(
            "LIBRARY_ALIAS_RESOLVED\t"
            f"archive_path={archive}\tmember=runtime/liburma.so\t"
            "loader_key=liburma.so\ttarget=runtime/liburma.so.0\t",
            result.stdout,
        )
        self.assertIn("\tmember=runtime/liburma.so.0\t", result.stdout)
        self.assertIn("\tlibraries=1\t", result.stdout)

    def test_library_alias_loader_key_participates_in_candidate_conflicts(
        self,
    ) -> None:
        foreign = self.write_archive_with_symlink(
            "foreign",
            "foreign.tar.gz",
            "libkvaf_other.so",
            "libkvaf_rt_aaaaaaaaaaaaaaaa.so",
            b"D-foreign-target",
        )
        self.registry.write_text(
            "name:foreign.tar.gz\npackage_path:foreign\n",
            encoding="utf-8",
        )
        result = self.run_audit(
            "--candidate-soname",
            "libkvaf_rt_aaaaaaaaaaaaaaaa.so="
            + hashlib.sha256(b"A-candidate").hexdigest(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "CANDIDATE_CONFLICT\t"
            "runtime_soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so",
            result.stderr,
        )
        self.assertIn(f"archive_path={foreign}", result.stderr)

    def test_regular_member_basename_participates_in_candidate_conflicts(
        self,
    ) -> None:
        foreign = self.write_archive(
            "foreign",
            "foreign.tar.gz",
            {"libkvaf_rt_aaaaaaaaaaaaaaaa.so": b"C-foreign-soname"},
        )
        self.registry.write_text(
            "name:foreign.tar.gz\npackage_path:foreign\n",
            encoding="utf-8",
        )

        result = self.run_audit(
            "--candidate-soname",
            "libkvaf_rt_aaaaaaaaaaaaaaaa.so="
            + hashlib.sha256(b"A-candidate").hexdigest(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "CANDIDATE_CONFLICT\t"
            "runtime_soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so",
            result.stderr,
        )
        self.assertIn(f"archive_path={foreign}", result.stderr)

    def test_multihop_library_alias_chain_resolves_to_regular_elf(self) -> None:
        archive = self.write_archive_with_symlink_chain(
            "packages",
            "official.tar.gz",
            "runtime/liburma.so.1.2",
            [
                ("runtime/liburma.so.1", "liburma.so.1.2"),
                ("runtime/liburma.so", "liburma.so.1"),
            ],
            b"C-official-chain",
        )
        self.registry.write_text(
            "name:official.tar.gz\npackage_path:packages\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            f"archive_path={archive}\tmember=runtime/liburma.so\t"
            "loader_key=liburma.so\ttarget=runtime/liburma.so.1\t",
            result.stdout,
        )
        self.assertIn(
            "resolved_target=runtime/liburma.so.1.2", result.stdout
        )

    def test_replaced_archive_is_excluded_from_global_conflict_set(
        self,
    ) -> None:
        current = self.write_archive(
            "current",
            "current.tar.gz",
            {"libkvaf_rt_aaaaaaaaaaaaaaaa.so": b"A-old"},
        )
        self.write_archive(
            "foreign",
            "foreign.tar.gz",
            {"libkvaf_rt_aaaaaaaaaaaaaaaa.so": b"B-foreign"},
        )
        self.registry.write_text(
            "name:current.tar.gz\npackage_path:current\n"
            "name:foreign.tar.gz\npackage_path:foreign\n",
            encoding="utf-8",
        )
        new_soname = "libkvaf_rt_cccccccccccccccc.so"
        new_sha = hashlib.sha256(b"C-new").hexdigest()

        result = self.run_audit(
            "--replace-archive",
            str(current),
            "--candidate-soname",
            f"{new_soname}={new_sha}",
        )

        self.assert_success(result)
        self.assertIn("\tconflicts=0\tcandidate_conflicts=0\t", result.stdout)

    def test_duplicate_identical_runtime_soname_and_sha_is_not_conflict(
        self,
    ) -> None:
        content = b"A-identical-provider"
        self.write_archive(
            "one",
            "one.tar.gz",
            {"libkvaf_rt_aaaaaaaaaaaaaaaa.so": content},
        )
        self.write_archive(
            "two",
            "two.tar.gz",
            {"nested/libkvaf_rt_aaaaaaaaaaaaaaaa.so": content},
        )
        self.registry.write_text(
            "name:one.tar.gz\npackage_path:one\n"
            "name:two.tar.gz\npackage_path:two\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            "\tconflicts=0\tcandidate_conflicts=0\terrors=0",
            result.stdout,
        )
        self.assertNotIn("CONFLICT", result.stderr)

    def test_unsafe_package_path_is_rejected_before_archive_access(self) -> None:
        self.registry.write_text(
            "name:escape.tar.gz\npackage_path:../outside\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe package_path", result.stderr)
        self.assertNotIn("STATIC_AUDIT", result.stdout)

    def test_missing_registered_archive_fails_closed(self) -> None:
        self.registry.write_text(
            "name:missing.tar.gz\npackage_path:packages\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("registered archive is missing or unreadable", result.stderr)
        self.assertIn(
            "SUMMARY\tregistered_packages=1\tlibraries=0\t"
            "conflicts=0\tcandidate_conflicts=0\terrors=1",
            result.stdout,
        )

    def test_missing_optional_official_archive_is_reported_not_failed(
        self,
    ) -> None:
        missing = self.root / "opp" / "Ascend-aicpu_legacy.tar.gz"
        self.registry.write_text(
            "name:Ascend-aicpu_legacy.tar.gz\n"
            "package_path:opp\n"
            "optional:true\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            f"PACKAGE_UNAVAILABLE\tarchive_path={missing}\t"
            "package_path=opp\tname=Ascend-aicpu_legacy.tar.gz\t"
            "optional=true\treason=optional-missing",
            result.stdout,
        )
        self.assertIn(
            "\terrors=0\toptional_missing=1\topaque_packages=0\t"
            "unaudited_packages=0\tcandidate_unknowns=0\tcoverage=partial",
            result.stdout,
        )
        self.assertEqual(result.stderr, "")

    def test_missing_optional_kvaf_target_still_fails_closed(self) -> None:
        self.registry.write_text(
            "name:cann-kvaf-direct-urma-compat.tar.gz\n"
            "package_path:packages\n"
            "optional:true\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("registered archive is missing or unreadable", result.stderr)
        self.assertNotIn("PACKAGE_UNAVAILABLE", result.stdout)

    def test_plain_tar_with_gzip_suffix_is_inspected_by_content(self) -> None:
        archive = self.write_plain_tar_with_gzip_suffix(
            "built-in",
            "official.tar.gz",
            {"runtime/liburma.so.0": b"C-plain-tar"},
        )
        self.registry.write_text(
            "name:official.tar.gz\npackage_path:built-in\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            f"PACKAGE_CONTAINER\tarchive_path={archive}\t"
            "container_hint=tar\taudit_format=tar-auto",
            result.stdout,
        )
        self.assertIn("\tmember=runtime/liburma.so.0\t", result.stdout)
        self.assertIn("\tcoverage=complete", result.stdout)

    def test_opaque_official_package_is_explicit_partial_coverage(
        self,
    ) -> None:
        archive = self.root / "built-in" / "aicpu_hcomm.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"CANN-private-container\x00payload")
        self.registry.write_text(
            "name:aicpu_hcomm.tar.gz\npackage_path:built-in\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(
            f"PACKAGE_UNAUDITED\tarchive_path={archive}\t",
            result.stdout,
        )
        self.assertIn(
            "\tname=aicpu_hcomm.tar.gz\tcontainer_hint=opaque\t"
            "reason=vendor-private-or-unsupported-container",
            result.stdout,
        )
        self.assertIn(
            "\terrors=0\toptional_missing=0\topaque_packages=1\t"
            "unaudited_packages=1\tcandidate_unknowns=0\tcoverage=partial",
            result.stdout,
        )
        self.assertNotIn("CONFLICT", result.stderr)

    def test_candidate_check_fails_on_opaque_scope_without_false_conflict(
        self,
    ) -> None:
        archive = self.root / "built-in" / "aicpu_hcomm.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"CANN-private-container\x00payload")
        self.registry.write_text(
            "name:aicpu_hcomm.tar.gz\npackage_path:built-in\n",
            encoding="utf-8",
        )

        result = self.run_audit(
            "--candidate-soname",
            "libkvaf_candidate.so=" + hashlib.sha256(b"candidate").hexdigest(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            f"CANDIDATE_UNAUDITED_SCOPE\tarchive_path={archive}\t",
            result.stderr,
        )
        self.assertNotIn("CANDIDATE_CONFLICT\t", result.stderr)
        self.assertIn("\tcandidate_unknowns=1\tcoverage=partial", result.stdout)

    def test_opaque_kvaf_target_is_a_strict_error(self) -> None:
        archive = self.root / "cust" / "cann-kvaf-test.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"opaque-kvaf-container")
        self.registry.write_text(
            "name:cann-kvaf-test.tar.gz\npackage_path:cust\noptional:true\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "strict KVAF archive is not a readable tar container", result.stderr
        )
        self.assertNotIn("PACKAGE_UNAUDITED", result.stdout)

    def test_readable_kvaf_target_without_runtime_library_is_an_error(
        self,
    ) -> None:
        self.write_archive(
            "cust",
            "cann-kvaf-empty.tar.gz",
            {"aicpu_kernels_device/manifest.txt": b"metadata-only"},
        )
        self.registry.write_text(
            "name:cann-kvaf-empty.tar.gz\npackage_path:cust\noptional:true\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "strict KVAF archive contains no inspectable runtime library",
            result.stderr,
        )

    def test_dot_package_path_is_canonically_audited(self) -> None:
        archive = self.write_archive(
            ".", "root-package.tar.gz", {"lib/liburma.so.0": b"C-root"}
        )
        self.registry.write_text(
            "name:root-package.tar.gz\npackage_path:.\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assert_success(result)
        self.assertIn(f"PACKAGE\tarchive_path={archive}", result.stdout)

    def test_registered_archive_symlink_is_rejected(self) -> None:
        real = self.write_archive(
            "real", "real.tar.gz", {"lib/liburma.so.0": b"C-real"}
        )
        linked_dir = self.root / "linked"
        linked_dir.mkdir()
        (linked_dir / "linked.tar.gz").symlink_to(real)
        self.registry.write_text(
            "name:linked.tar.gz\npackage_path:linked\n",
            encoding="utf-8",
        )

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("registered archive path is a symlink", result.stderr)

    def test_candidate_rejects_any_foreign_registration_and_ignores_replaced_archive(
        self,
    ) -> None:
        content = b"A-current-provider"
        current = self.write_archive(
            "current",
            "current.tar.gz",
            {"libkvaf_rt_aaaaaaaaaaaaaaaa.so": content},
        )
        foreign = self.write_archive(
            "foreign",
            "foreign.tar.gz",
            {"nested/libkvaf_rt_aaaaaaaaaaaaaaaa.so": content},
        )
        self.registry.write_text(
            "name:current.tar.gz\npackage_path:current\n"
            "name:foreign.tar.gz\npackage_path:foreign\n",
            encoding="utf-8",
        )

        binding = (
            "libkvaf_rt_aaaaaaaaaaaaaaaa.so="
            + hashlib.sha256(content).hexdigest()
        )
        result = self.run_audit(
            "--replace-archive", str(current),
            "--candidate-soname", binding,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "CANDIDATE_CONFLICT\t"
            "runtime_soname=libkvaf_rt_aaaaaaaaaaaaaaaa.so",
            result.stderr,
        )
        self.assertIn(f"archive_path={foreign}", result.stderr)
        self.assertNotIn(f"archive_path={current}\tmember=", result.stderr)
        self.assertIn("\tcandidate_conflicts=1\terrors=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
