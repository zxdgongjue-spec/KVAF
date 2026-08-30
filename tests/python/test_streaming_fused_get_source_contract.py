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
ACCESS_OPERATOR = ROOT / "access_operator/src/access_operator.cc"


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


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

    # A terminal Direct-HCCP CQ does not make a queue reusable by itself: the
    # owner-table CAS is the final ownership proof.  The helper must retain
    # the nonzero owner word when the CAS fails, and the cold Runtime teardown
    # must reject that table rather than releasing HCCP resources underneath
    # it.  This is source-only because injecting a CAS mismatch into a real
    # HCCP queue would intentionally corrupt the ownership protocol.
    release = _between(
        text,
        "__simt_callee__ inline bool ReleaseKvafStreamingBatchGetQueueFromSimt(",
        "__simt_vf__ __aicore__ LAUNCH_BOUND(1)\ninline void KvafStreamingBatchGetReleaseSimtService",
    )
    assert "asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0)" in release
    assert "return false;" in release
    assert "lease->owner_tag = 0;" in release
    assert release.index("return false;") < release.index("lease->owner_tag = 0;")

    access_operator = ACCESS_OPERATOR.read_text()
    teardown = _between(
        access_operator,
        "Status DefaultAccessOperator::ReleaseStreamingBatchGetControl()",
        "Status DefaultAccessOperator::DescribeStreamingBatchGetControl(",
    )
    for token in (
        "std::vector<std::int32_t> lease_snapshot",
        "aclrtSynchronizeDevice()",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "streaming BatchGet control teardown is quarantined",
        "StatusCode::kBusy",
    ):
        assert token in teardown, token
    assert teardown.index("lease_snapshot") < teardown.index(
        "KvafBatchGetLocalBufferRelease"
    )
    assert teardown.index("aclrtSynchronizeDevice()") < teardown.index(
        "ACL_MEMCPY_DEVICE_TO_HOST"
    )

    print(
        "KVAF_STREAMING_FUSED_GET_SOURCE_CONTRACT_PASS "
        "global_abi_aliases=1 gm_local_fieldwise_copy=1 "
        "direct_hccp_abi_layout=unchanged "
        "terminal_release_cas_quarantine=1"
    )


if __name__ == "__main__":
    main()
