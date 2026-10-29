"""Check standalone AIV ELF metadata without loading the kernel or needing CANN."""
from pathlib import Path
import struct


def check_aiv_kernel_metadata(path: str, kernel: str) -> None:
    """Require the SDK's per-function KTYPE TLV (uint16, uint16, uint32)."""
    data = Path(path).read_bytes()

    def span(offset, size):
        if offset < 0 or size < 0 or offset + size > len(data):
            raise ValueError("truncated ELF")
        return data[offset:offset + size]

    ident = span(0, 16)
    if ident[:4] != b"\x7fELF" or ident[5:7] != b"\x01\x01":
        raise ValueError("expected a little-endian ELF kernel")
    if ident[4] == 2:
        header_format, section_format = "<16sHHIQQQIHHHHHH", "<IIQQQQIIQQ"
    elif ident[4] == 1:
        header_format, section_format = "<16sHHIIIIIHHHHHH", "<IIIIIIIIII"
    else:
        raise ValueError("unsupported ELF class")
    header = struct.unpack(header_format, span(0, struct.calcsize(header_format)))
    table_offset, entry_size, count, names_index = header[6], header[11], header[12], header[13]
    section_size = struct.calcsize(section_format)
    if not table_offset or entry_size < section_size:
        raise ValueError("missing ELF section table")

    def section(index):
        return struct.unpack(section_format, span(table_offset + index * entry_size, section_size))

    # ELF extended section numbering stores these values in section zero.
    if not count:
        count = section(0)[5]
    if names_index == 0xffff:
        names_index = section(0)[6]
    if not count or not 0 < names_index < count:
        raise ValueError("invalid ELF section-name table")
    span(table_offset, count * entry_size)
    names_section = section(names_index)
    if names_section[1] != 3:  # SHT_STRTAB
        raise ValueError("invalid ELF section-name table type")
    names = span(names_section[4], names_section[5])
    expected = (".ascend.meta." + kernel).encode("ascii")
    matches = []
    for index in range(count):
        item = section(index)
        name_offset = item[0]
        end = names.find(b"\0", name_offset)
        if name_offset >= len(names) or end < 0:
            raise ValueError("invalid ELF section name")
        if names[name_offset:end] == expected:
            if item[1] != 1:  # SHT_PROGBITS
                raise ValueError("kernel metadata is not a data section")
            matches.append(span(item[4], item[5]))
    if len(matches) != 1:
        raise ValueError(f"expected one {expected.decode()} section, found {len(matches)}")

    # SDK utils/kernel_utils_macros.h: F_TYPE_KTYPE=1, K_TYPE_AIV=3.
    payload, offset, types = matches[0], 0, []
    while offset < len(payload):
        if len(payload) - offset < 4:
            raise ValueError("truncated kernel metadata TLV header")
        tag, size = struct.unpack_from("<HH", payload, offset)
        offset += 4
        if size > len(payload) - offset:
            raise ValueError("truncated kernel metadata TLV value")
        if tag == 1:
            if size != 4:
                raise ValueError("invalid KTYPE TLV size")
            types.append(struct.unpack_from("<I", payload, offset)[0])
        offset += size
    if types != [3]:
        raise ValueError(f"expected exactly one K_TYPE_AIV(3), got {types}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--kernel", default="KvafScatterBatchGetAivKernelV1")
    args = parser.parse_args()
    try:
        check_aiv_kernel_metadata(args.binary, args.kernel)
    except (OSError, ValueError) as error:
        parser.exit(1, f"AIV_METADATA_INVALID: {error}\n")
    print(f"AIV_METADATA_OK kernel={args.kernel} type=K_TYPE_AIV(3) binary={args.binary}")
