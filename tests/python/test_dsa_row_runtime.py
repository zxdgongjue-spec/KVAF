#!/usr/bin/env python3
"""CPU reference coverage for the DSA_ROW public Runtime path."""

import unittest

import kvaf


TOKENS_PER_BLOCK = 4
TOKEN_BYTES = 3
BLOCK_BYTES = TOKENS_PER_BLOCK * TOKEN_BYTES


def block_with_tokens(start: int) -> bytearray:
    """Build one block whose each fixed-width token is easy to identify."""
    return bytearray(
        byte for token in range(start, start + TOKENS_PER_BLOCK)
        for byte in (token, token + 1, token + 2)
    )


class DsaRowMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = kvaf.DsaRowMockRuntime()
        # 2 rows * 2 layers * 4 q blocks, with enough physical DDR blocks for
        # all cells.  The reference API has no KeyId/hash input anywhere.
        self.runtime.initialize(
            ddr_bytes=16 * 12,
            raw_row_capacity=2,
            n_layers=2,
            q_capacity=4,
            tokens_per_logical_block=TOKENS_PER_BLOCK,
            packed_token_bytes=TOKEN_BYTES,
            ddr_block_alignment_bytes=4,
        )
        self.assertEqual(BLOCK_BYTES, self.runtime.geometry["block_payload_bytes"])
        self.assertEqual(12, self.runtime.geometry["ddr_block_bytes"])
        self.runtime.bind_row(0, max_sequence_tokens=16)

    def tearDown(self) -> None:
        if self.runtime.initialized:
            # M0 intentionally refuses Runtime shutdown while a row remains
            # OPEN. Tests may stop after an expected Get validation failure,
            # so close the test-owned row first.
            try:
                self.runtime.close_row(0)
            except ValueError:
                pass
            self.runtime.shutdown()

    def test_full_block_commit_direct_get_and_close(self) -> None:
        layer0 = block_with_tokens(0) + block_with_tokens(4)
        layer1 = block_with_tokens(40) + block_with_tokens(44)

        committed = self.runtime.put_blocks(0, 2, [layer0, layer1])
        self.assertEqual("committed", committed["state"])
        self.assertEqual(4, committed["published_block_count"])
        self.assertEqual(2, committed["committed_block_count"])

        output = bytearray(3 * TOKEN_BYTES)
        got = self.runtime.get_tokens(0, 1, [0, 3, 4], [2, 0, 1], output)
        self.assertIs(got, output)
        self.assertEqual(bytes((43, 44, 45)), output[0:3])
        self.assertEqual(bytes((44, 45, 46)), output[3:6])
        self.assertEqual(bytes((40, 41, 42)), output[6:9])

        close = self.runtime.close_row(0)
        self.assertEqual("free", close["state"])
        self.assertEqual(4, close["released_block_count"])
        self.runtime.bind_row(0, max_sequence_tokens=16)
        self.assertEqual(0, self.runtime.close_row(0)["released_block_count"])

    def test_uncommitted_or_out_of_range_token_is_rejected(self) -> None:
        payload = block_with_tokens(0)
        self.runtime.put_blocks(0, 1, [payload, payload])
        with self.assertRaises(ValueError):
            self.runtime.get_tokens(0, 0, [4], [0], bytearray(TOKEN_BYTES))

    def test_default_c8_alignment(self) -> None:
        runtime = kvaf.DsaRowMockRuntime()
        block_payload = 128 * 656
        try:
            runtime.initialize(
                ddr_bytes=block_payload,
                raw_row_capacity=1,
                n_layers=1,
                q_capacity=1,
            )
            self.assertEqual(block_payload, runtime.geometry["block_payload_bytes"])
            self.assertEqual(block_payload, runtime.geometry["ddr_block_bytes"])
        finally:
            if runtime.initialized:
                runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
