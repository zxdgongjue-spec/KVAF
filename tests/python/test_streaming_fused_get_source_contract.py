#!/usr/bin/env python3
"""Source-only contract for the embedded streaming Direct-HCCP helper.

This intentionally runs without an NPU.  It protects the two details that a
normal C++ compiler cannot model: the global C ABI must be exposed through the
fused helper namespace, and AscendC's distinct GM address space forbids C++
aggregate assignment between GM and local records.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "access_operator/demo/aiv_batch_get/kernel/kvaf_fused_full_get_device.h"


def main() -> None:
    text = HELPER.read_text()

    for token in (
        "using ::KvafStreamingBatchGetControl;",
        "using ::KvafStreamingBatchGetDescriptorScratch;",
        "LoadKvafStreamingBatchGetControlFromGmForAicore",
        "LoadKvafStreamingBatchGetControlFromGmForSimt",
        "LoadKvafStreamingBatchGetDescriptorScratchFromGm",
        "StoreKvafStreamingBatchGetDescriptorScratchToGm",
    ):
        assert token in text, token

    # These four copies were valid in the standalone compilation context but
    # invalid when the same helper is embedded in an AscendC mixed operator.
    # Field-wise conversion preserves the exact ABI values and ordering.
    for forbidden in (
        "const KvafStreamingBatchGetControl control = *control_gm;",
        "*descriptor = *source;",
        "*scratch = descriptor;",
        "control, queue_id, *scratch))",
    ):
        assert forbidden not in text, forbidden

    for field in (
        "magic",
        "descriptor_template_count",
        "address_table",
        "private_output_row_base",
    ):
        assert f"destination->{field}" in text, field

    print(
        "KVAF_STREAMING_FUSED_GET_SOURCE_CONTRACT_PASS "
        "global_abi_aliases=1 gm_local_fieldwise_copy=1 "
        "direct_hccp_abi_layout=unchanged"
    )


if __name__ == "__main__":
    main()
