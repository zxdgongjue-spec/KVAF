"""Reject missing/wrong kernel types in the actual binary selected for launch."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from aiv_kernel_metadata import check_aiv_kernel_metadata
import aiv_batch_get_test_common as common

KERNEL = "KvafScatterBatchGetAivKernelV1"
AIV = struct.pack("<HHI", 1, 4, 3)
# PID 44853's 40-byte NOTE: a section exists, but no F_TYPE_KTYPE tag exists.
FAULT_METADATA = bytes.fromhex(
    "0c000400 01000000 07000400 00000000 "
    "08000400 70020000 09000400 00000000 "
    "0a000400 00000000")


def elf(payload=AIV, *, kernel=KERNEL, bits=64, extended=False,
        duplicate=False, section_type=7):
    header_format = "<16sHHIQQQIHHHHHH" if bits == 64 else "<16sHHIIIIIHHHHHH"
    section_format = "<IIQQQQIIQQ" if bits == 64 else "<IIIIIIIIII"
    header_size, section_size = struct.calcsize(header_format), struct.calcsize(section_format)
    count = 4 if duplicate else 3
    names = b"\0.shstrtab\0.ascend.meta." + kernel.encode() + b"\0"
    names_offset = header_size + count * section_size
    payload_offset = (names_offset + len(names) + 3) & ~3
    ident = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1, 1]) + b"\0" * 9
    header = struct.pack(header_format, ident, 1, 0, 1, 0, 0, header_size,
                         0, header_size, 0, 0, section_size, 0 if extended else count,
                         0xffff if extended else 1)
    null = (0, 0, 0, 0, 0, count if extended else 0, 1 if extended else 0, 0, 0, 0)
    strings = (1, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    metadata = (11, section_type, 0, 0, payload_offset, len(payload), 0, 0, 4, 0)
    sections = [null, strings, metadata] + ([metadata] if duplicate else [])
    result = header + b"".join(struct.pack(section_format, *s) for s in sections) + names
    return result + b"\0" * (payload_offset - len(result)) + payload


class AivKernelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.binary = Path(self.directory.name) / "kernel.o"

    def check(self, data):
        self.binary.write_bytes(data)
        check_aiv_kernel_metadata(str(self.binary), KERNEL)

    def test_aiv_in_note_and_progbits(self):
        for bits in (32, 64):
            for extended in (False, True):
                for section_type in (1, 7):
                    with self.subTest(bits=bits, extended=extended, section_type=section_type):
                        self.check(elf(bits=bits, extended=extended, section_type=section_type))

    def test_actual_fault_metadata_is_missing_type_not_bad_section(self):
        with self.assertRaisesRegex(ValueError, r"expected exactly one K_TYPE_AIV\(3\), got \[\]"):
            self.check(elf(FAULT_METADATA))

    def test_type_can_be_before_or_after_compiler_metadata(self):
        self.check(elf(AIV + FAULT_METADATA))
        self.check(elf(FAULT_METADATA + AIV))

    def test_missing_wrong_and_duplicate_type(self):
        for data in (elf(kernel="AnotherKernel"), elf(b""), elf(duplicate=True),
                     elf(struct.pack("<HHI", 1, 4, 2)), elf(AIV + AIV),
                     elf(struct.pack("<HHI", 2, 4, 3)), elf(section_type=8)):
            with self.subTest(data=data[-16:]), self.assertRaises(ValueError):
                self.check(data)

    def test_malformed_file_and_tlv(self):
        for data in (b"", b"not ELF" * 10, elf()[:-1], elf(b"\x01"),
                     elf(struct.pack("<HHI", 1, 8, 3)), elf(struct.pack("<HH", 1, 0))):
            with self.subTest(data=data[-16:]), self.assertRaises(ValueError):
                self.check(data)

    def test_invalid_binary_stops_before_hardware_import(self):
        self.binary.write_bytes(elf(FAULT_METADATA))
        with patch.dict("os.environ", {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_SCATTER_PROBE": "load"}), \
             patch.object(common, "kernel_paths", return_value=(str(self.binary), "unused-put.o")):
            with self.assertRaisesRegex(RuntimeError, "scatter AIV metadata check failed"):
                common.run("cpu_push_scatter")

    def test_valid_binary_reaches_hardware_setup(self):
        self.binary.write_bytes(elf(FAULT_METADATA + AIV))
        original_import = __import__

        def importing(name, *args, **kwargs):
            if name == "backend":
                raise RuntimeError("reached hardware setup")
            return original_import(name, *args, **kwargs)

        with patch.dict("os.environ", {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_SCATTER_PROBE": "load"}), \
             patch.object(common, "kernel_paths", return_value=(str(self.binary), "unused-put.o")), \
             patch("builtins.__import__", side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, "reached hardware setup"):
                common.run("cpu_push_scatter")


if __name__ == "__main__":
    unittest.main()
