"""Source and queue-accounting checks for the opt-in fused dense PostOnly.

These CPU checks do not substitute for Ascend C compilation or real HCCP
transfer. They guard the protocol properties that the caller's SQ/CQ ledger
assumes before the NPU A/B test runs.
"""

from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
).read_text()


def _section(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)]


def test_single_doorbell_is_opt_in_after_all_sq_copies() -> None:
    post = _section(
        "__aicore__ inline void PostDenseBatchGetRequestsSingleDoorbell(",
        "// Preserve the proven one-warp path",
    )
    assert post.count("st_dev(") == 1
    assert post.count("WriteGmByPassDCache<uint64_t>(") == 1
    assert post.count("set_flag(PIPE_MTE3, PIPE_S,") == 1
    assert post.count("wait_flag(PIPE_MTE3, PIPE_S,") == 1
    assert post.index("CopyDensePreparedPostToSq(final_post)") < post.index(
        "wait_flag(PIPE_MTE3, PIPE_S,"
    ) < post.index("WriteGmByPassDCache<uint64_t>(") < post.index(
        "DataSyncBarrier<MemDsbT::DDR>();"
    ) < post.index("st_dev(")
    assert "previous + kWqeBankCount < total_posts" in post
    assert "set_flag(PIPE_MTE3, PIPE_V," in post
    assert "post_index >= kWqeBankCount" in post
    assert "completion_on_last_wqe" in post
    assert "post_index + 1U == total_posts" in post

    scalar = _section(
        "__aicore__ inline uint64_t BeginKvafDenseBatchGetPostOnlyScalar(",
        "// Probe the one CQE",
    )
    assert scalar.index("ValidateDenseBatchGetDescriptors(") < scalar.index(
        "PrepareDenseBatchGetOutputForInboundDma"
    ) < scalar.index("PostDenseBatchGetRequestsSingleDoorbell(")
    assert "bool batched_doorbell = false" in scalar
    assert "if (batched_doorbell)" in scalar
    assert "PostDenseBatchGetRequests(" in scalar
    assert "BeginKvafDenseBatchGetPostOnlyScalarBatchedDoorbell(" in scalar


@pytest.mark.parametrize("count", [1, 31, 32, 33, 64, 127, 128, 255, 256])
@pytest.mark.parametrize("head", [0, 1, 255, 511, 512, 1023])
def test_dense_wave_has_one_signaled_completion_and_wrap_safe_sq(
    count: int, head: int
) -> None:
    depth = 512
    # Model the same bank and PI equations as FillDenseBatchGetReadWqes.
    posts = (count + 31) // 32
    produced = []
    bank_release = []
    for post in range(posts):
        offset = post * 32
        bank = post % 2
        if post >= 2:
            assert bank_release[post - 2] == bank
        rows = min(32, count - offset)
        produced.extend(
            (
                head + offset + slot,
                (head + offset + slot) % depth,
                1 - (((head + offset + slot) // depth) & 1),
                post + 1 == posts and slot + 1 == rows,
            )
            for slot in range(rows)
        )
        bank_release.append(bank)

    assert len(produced) == count
    assert [pi for pi, _, _, _ in produced] == list(range(head, head + count))
    assert sum(signaled for _, _, _, signaled in produced) == 1
    assert produced[-1][3]
    assert produced[-1][2] == 1 - (((head + count - 1) // depth) & 1)
    # One PI publication advances by all rows; only one terminal CQE is due.
    assert head + count - head == count
    expected_cq = 1
    assert expected_cq == sum(signaled for _, _, _, signaled in produced)


def test_failure_and_zero_count_do_not_enter_single_doorbell_path() -> None:
    scalar = _section(
        "__aicore__ inline uint64_t BeginKvafDenseBatchGetPostOnlyScalar(",
        "// Probe the one CQE",
    )
    assert "active_request_count == 0U" in scalar
    assert "active_request_count > args.request_capacity" in scalar
    assert "args.request_capacity > kDenseRequestCapacity" in scalar
    assert "result == kSuccess &&\n      ActiveRequestsForJetty(" in scalar
    assert scalar.index("if (batched_doorbell) {") > scalar.index(
        "ValidateDenseBatchGetDescriptors("
    )
