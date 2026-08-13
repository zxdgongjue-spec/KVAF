# DSA_ROW M0 Reference Runtime

## Scope

This change introduces the M0 row-major control-plane contract used by the
CacheOS Stage 2 W4A4C8 DSA design.  It deliberately does **not** enable a
vLLM serving path yet.

The in-tree implementation is a synchronous CPU reference runtime.  It
validates the fixed directory and lifecycle rules before the future A5 data
plane is connected:

- packed layout: 128 tokens per logical block and 656 bytes per packed token;
- `flat_index = pool_row * (n_layers * q_capacity) + layer * q_capacity + block_q`;
- no `KeyId`, key hash, or legacy `AccessTable` on the DSA_ROW path;
- full-layer Group Put: Prepare -> Write -> Seal -> Commit / Abort;
- sparse-row read pins and all-or-nothing Direct Get validation;
- Close only after all Put/Get/access ownership is settled; unexpected state
  quarantines the row instead of returning DDR blocks to the free list.

`DsaRowMockRuntime` is intentionally named as a mock.  It is CPU-only and
advertises `cpu_reference_only`; it must not be used as
`kvaf.DsaRowRuntime` by an online vLLM process.

## Build and test

Run from the KVAF repository root.  These commands are suitable for the
Docker environment on server B as well as a development host with CMake and
the selected Python interpreter available.

```bash
PY=/usr/local/python3.11.10/bin/python

# Optional but needed when KVAF_BUILD_PYTHON=ON.
$PY -m pybind11 --cmakedir

cmake -S . -B build/dsa-row-m0 \
  -DBUILD_TESTING=ON \
  -DKVAF_BUILD_PYTHON=ON
cmake --build build/dsa-row-m0 -j"$(nproc)"
ctest --test-dir build/dsa-row-m0 --output-on-failure
```

The Python test can also be executed directly after a successful build:

```bash
PYTHONPATH="$PWD/build/dsa-row-m0/python" \
  "$PY" tests/python/test_dsa_row_runtime.py
```

The C++ test covers the real C8 geometry
`128 * 656 = 83968` bytes with a 32-byte DDR alignment, as well as
Dense/PAD Get rejection, whole-batch invalid-Get rejection, read-pin/Close
ordering, and Shutdown refusal while rows remain open.

## Production gate

Before `storage_backend="kvaf_dsa_row"` can serve requests, KVAF must expose
a production `kvaf.DsaRowRuntime` that advertises
`probe_dsa_row_payload_abi() == "torch_npu_packed_v1"` and implements the
full NPU row API.  It must bind packed NPU resident caches, publish dirty
device-directory ranges, and route C8 full-block Put plus 656-byte Direct Get
through the A5 transport.  vLLM rejects the CPU mock and refuses mixed
legacy/KVAF payload paths by design.
