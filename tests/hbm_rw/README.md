# CPU -> NPU HBM read/write test

This is an isolated two-process hardware test. It does not use `RuntimeCore`,
the KV metadata path, or the production `PeerLink` protocol.

Build on a host that has both CANN and UMDK/liburma:

```bash
./build.sh hbm-rw --build-dir build-hbm-rw
```

On the NPU host, the server discovers `remote_eid` from the selected
`/proc/asdrv_ub/pair_info` block, enumerates `RaGetDevEidInfoList`, and uses
the matching list entry as the local `RaCtxInit` EID:

```bash
tests/run_hbm_rw_server.sh build-hbm-rw \
  --device-id 0 --eid-index 0 --port 29888 --bytes 4096
```

`--eid-index` is the context EID index. For diagnostics, bypass pair-info
matching with `--npu-eid-list-index N`; `--eid HEX` remains an explicit EID
override.

On the CPU host, `local_eid` is read from `pair_info`; pass `--host-eid HEX`
only when overriding discovery:

```bash
tests/run_hbm_rw_client.sh build-hbm-rw <npu-host> 29888 \
  --bytes 4096 --device-id 0 --eid-index 0
```

The client registers a local host buffer, imports the NPU HBM segment, creates
the CTP/Jetty pair, performs one CPU-originated WRITE and one READ, then checks
CQ completion and byte equality. The server keeps the HBM registration alive
until the client has completed the transfers.

The test accepts explicit EIDs for diagnostics, but the default path follows
the pair-info local/remote EID rule used by the H2D script.
