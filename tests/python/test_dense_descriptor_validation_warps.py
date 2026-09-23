"""Guard the small-window and bounded parallel pre-Post validation routes."""

from pathlib import Path


KERNEL = (
    Path(__file__).resolve().parents[2]
    / "access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
)


def test_dense_descriptor_validation_keeps_one_warp_for_small_waves():
    source = KERNEL.read_text()
    wrapper = source.split("__aicore__ inline void ValidateDenseBatchGetDescriptors(", 1)[1]
    wrapper = wrapper.split("__aicore__ inline uint32_t ActiveRequestsForJetty(", 1)[0]
    assert "if (active_request_count <= 48U)" in wrapper
    assert "VF_CALL<ValidateDenseBatchGetDescriptorsSmall>" in wrapper
    assert "*result = failure[1];" in wrapper
    assert "return;" in wrapper


def test_dense_descriptor_validation_preserves_error_order_and_bounds():
    source = KERNEL.read_text()
    large = source.split("inline void ValidateDenseBatchGetDescriptorsSimt(", 1)[1]
    large = large.split("__aicore__ inline void ValidateDenseBatchGetDescriptors(", 1)[0]
    for check in (
        "table_capacity", "valid_table", "kAddressTableEntryBytes",
        "output_rows", "local_hbm_size", "remote_host_ddr_size",
        "first_failure", "asc_shfl_down", "failure[warp * 2U]",
    ):
        assert check in large
    wrapper = source.split("__aicore__ inline void ValidateDenseBatchGetDescriptors(", 1)[1]
    wrapper = wrapper.split("__aicore__ inline uint32_t ActiveRequestsForJetty(", 1)[0]
    assert "active_request_count <= 64U ? 64U" in wrapper
    assert "active_request_count <= 128U ? 128U : 256U" in wrapper
    assert "candidate < first_failure" in wrapper
    assert "candidate = failure[warp * 2U]" in wrapper
