# NPU UB port traffic

Use `npu_port_stat.py` alongside BatchGet to sample A5 NPU UB port TX/RX
bandwidth through `hccn_tool`, with optional CTP packet rates and raw flit rates.
See [NPU port sampling](npu_port_stat.md) for device discovery, commands and
measurement scope. CPU-side `ubctl` chip/controller IDs are not NPU device IDs.

# CPU-side URMA information and port/VL bandwidth

Run both tools on the CPU server. Use `collect_read_qos_info.py` once per
environment/configuration to collect local URMA/UBASE driver identity, EID-bound
QoS configuration and optional CPU driver source excerpts in `report.json` and
raw files. Use `collect_read_qos.py` for each workload to sample one or more devices
concurrently, pairing `--urma-device` and `--eid` values in order. Raw `perf_*.txt`
and timing/status go under `<output>/<device>/`; the root `sampling.log` records
overall status. Devices sharing a controller reuse the same samples. Neither tool
queries NPU/HCCP/Ascend or live Jetty/TP resources. Sampling does not produce a
JSON report or bandwidth CSV.
See [CPU information and port/VL sampling](read_qos.md) for commands and output
details. The old `--role cpu` option is no longer needed or accepted.

# Paired bandwidth timing scopes

Run from KVAF on ak-batchget-test, using its matching build-npu. Existing
benchmark and kernel files are not modified. The runner instruments a copy of
the benchmark source in memory and refuses unknown source layouts.

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/bandwidth_scopes.py \
  --build build-npu --output bandwidth-5000.json --clock-mhz 1000 \
  --token-bytes 656 --request-length 32768 --miss-count 5000 \
  --aiv-cores 2 --jettys-per-core 1 --warmup 3 --iterations 30
```

Repeat with miss-count 2048 and 8192, using different output names. Keep the
same machine, runtime build and network configuration for V9. Clock MHz is an
explicit assumption, not autodetection; retain raw ticks and confirm frequency.

Each JSON sample pairs host elapsed, external NPU event elapsed, kernel wall
elapsed, and first-doorbell-to-last-completion elapsed with payload bytes.
Host/event instrumentation itself has overhead. Kernel wall uses cross-core
min/max timestamps, so it requires a common clock. Transfer overlaps posting:
do not add those durations. Original correctness preflight remains active.
The independent upstream script remains the uninstrumented control.

Validation: python3 -m unittest discover -s tests/diagnostics -p 'test_*.py'.
Actual device execution must be validated on an NPU.
