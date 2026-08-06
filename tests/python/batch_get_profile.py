"""Formatting helpers for opt-in NPU BatchGet host timing profiles."""

from __future__ import annotations

import statistics


def print_batch_get_profile(
    profile: dict[str, object],
    total_samples_ms: list[float],
    submit_samples_ms: list[float],
    synchronize_samples_ms: list[float],
) -> None:
    mean_us = profile["mean_us"]
    assert isinstance(mean_us, dict)
    samples = int(profile["samples"])
    python_total_us = statistics.mean(total_samples_ms) * 1000.0
    python_submit_us = statistics.mean(submit_samples_ms) * 1000.0
    python_synchronize_us = statistics.mean(synchronize_samples_ms) * 1000.0
    binding_total_us = float(mean_us["binding_total"])
    print(
        "BATCH_GET_PROFILE level=python "
        f"samples={samples} total_us={python_total_us:.3f} "
        f"submit_us={python_submit_us:.3f} "
        f"synchronize_us={python_synchronize_us:.3f} "
        f"pybind_boundary_us={max(0.0, python_submit_us - binding_total_us):.3f}"
    )
    print(
        "BATCH_GET_PROFILE level=binding "
        f"samples={samples} total_us={binding_total_us:.3f} "
        f"tensor_views_us={float(mean_us['binding_tensor_views']):.3f} "
        f"validate_build_us={float(mean_us['binding_validate_and_build_request']):.3f} "
        f"resolve_stream_us={float(mean_us['binding_resolve_stream']):.3f} "
        f"runtime_call_us={float(mean_us['binding_runtime_call']):.3f} "
        f"unaccounted_us={float(mean_us['binding_unaccounted']):.3f}"
    )
    print(
        "BATCH_GET_PROFILE level=tensor_views "
        f"samples={samples} indices_us={float(mean_us['binding_indices_tensor']):.3f} "
        f"mask_us={float(mean_us['binding_mask_tensor']):.3f} "
        f"output_indices_us={float(mean_us['binding_output_indices_tensor']):.3f} "
        f"output_us={float(mean_us['binding_output_tensor']):.3f} "
        f"table_addresses_us={float(mean_us['binding_table_addresses_tensor']):.3f} "
        f"table_valid_us={float(mean_us['binding_table_valid_tensor']):.3f}"
    )
    print(
        "BATCH_GET_PROFILE level=runtime "
        f"samples={samples} total_us={float(mean_us['runtime_total']):.3f} "
        f"validation_us={float(mean_us['runtime_validation']):.3f} "
        f"operator_us={float(mean_us['runtime_operator']):.3f} "
        f"unaccounted_us={float(mean_us['runtime_unaccounted']):.3f}"
    )
    print(
        "BATCH_GET_PROFILE level=operator "
        f"samples={samples} total_us={float(mean_us['operator_total']):.3f} "
        f"build_args_us={float(mean_us['operator_build_args']):.3f} "
        f"acquire_us={float(mean_us['operator_local_buffer_acquire']):.3f} "
        f"launch_us={float(mean_us['operator_kernel_launch']):.3f} "
        f"internal_sync_us={float(mean_us['operator_stream_synchronize']):.3f} "
        f"status_copy_us={float(mean_us['operator_status_copy']):.3f} "
        f"release_us={float(mean_us['operator_local_buffer_release']):.3f} "
        f"unaccounted_us={float(mean_us['operator_unaccounted']):.3f}"
    )
