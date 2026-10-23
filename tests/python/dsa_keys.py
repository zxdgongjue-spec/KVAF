"""DSA directory keys shared by the BatchGet test setup paths."""

from __future__ import annotations


def build_dsa_keys(
    count: int, *, q_capacity: int, start: int = 0,
    pool_rows: int = 1, n_layers: int = 1, layer: int = 0,
) -> list[int]:
    """Encode a slice of one layer's blocks in pool-row/block-q order.

    start is a block ordinal, not a KeyId. The existing bandwidth scenarios
    use one pool row and one layer, where the DSA formula reduces to block_q.
    """
    if min(pool_rows, n_layers, q_capacity) <= 0 or not 0 <= layer < n_layers:
        raise ValueError("invalid DSA geometry or layer ordinal")
    if pool_rows * n_layers * q_capacity > (1 << 31) - 1:
        raise ValueError("DSA directory exceeds its int32 index range")
    if start < 0 or count < 0 or start + count > pool_rows * q_capacity:
        raise ValueError("DSA block range exceeds the selected layer's capacity")
    keys = []
    for ordinal in range(start, start + count):
        pool_row, block_q = divmod(ordinal, q_capacity)
        keys.append(pool_row * (n_layers * q_capacity) + layer * q_capacity + block_q)
    return keys
