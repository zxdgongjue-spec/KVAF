These two files are byte-for-byte copies from commit
`a12bb16c120e885eb18ed61581739e1c212c32e8` on `codex/aicpu-ordered-write`:

| File | Original path | Git blob |
| --- | --- | --- |
| abi.h | tests/aicpu_batch_get/abi.h | f71cf36671024e06c941ed6bb1a33b4a5d8c942b |
| pipeline.h | tests/aicpu_batch_get/pipeline.h | 15fdb12ef027f71e3bb5835aef09bb377b78d01f |

The production CPU-push path calls this V24 chunk/ready loop and per-row copy
through `cpu_push_pipeline.h`. Keep these snapshots unchanged so differential
tests and the migration audit have an exact source baseline.

Runtime-specific adaptations remain outside this directory: dynamic Tensor
source resolution, output-index lookup, profile results, one launch per call,
and explicit remote error completion codes. CPU WRITE chains retain four
Jettys, 1024-row chunks, up to 16 local SGEs clamped to device limits, 32-WR
doorbells, ordered ready markers and tail-only CQEs.
