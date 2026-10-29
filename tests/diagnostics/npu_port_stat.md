# BatchGet 的 NPU UB 端口采集

`npu_port_stat.py` 在 NPU 所在机器运行，通过 A5 版本的 `hccn_tool`
读取 NPU 的 UB 物理端口。无需改动或重编 BatchGet。

## 先确认设备和端口

以下以 hccn_tool 的 NPU logic ID 7 为例；按本轮实际设备修改。
容器或可见设备重映射时，先确认这个 ID 与测试使用的 NPU 对应，不能直接沿用
CPU `ubctl -c` 的 chip ID，也不能将 `ubctl -d` 当作 NPU ID。

```bash
python3 tests/diagnostics/npu_port_stat.py --device-id 7 --list
```

脚本查询 `hccn_tool -g -dev_info -i 7`，识别 `UdieID / PortID / PortType /
Link Status` 表，只采集 `PortType=UB` 的端口。默认选择其中状态为 UP 的端口。
使用本机安装的 hccn_tool；可通过 `--hccn-tool /实际路径/hccn_tool` 指定。
命令需要有读取 NPU 诊断信息的权限；失败时会显示错误。

## 测试期间采集

保持原来的 CPU 服务地址、NPU EID、peer UB device 和测试参数。
为覆盖采样窗口，可增加现有性能测试的 `--iterations`。以未重映射的 NPU7 为例：

```bash
PYTHONUNBUFFERED=1 DEVICE_ID=7 \
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --iterations 100000 --no-kernel-profile
```

上述示例的其余参数采用测试默认值；复现已有测试时应补回该测试的参数。
等日志出现 `BATCH_GET_PERF_PHASE_BEGIN phase=batch_get` 后，在另一终端运行：

```bash
python3 tests/diagnostics/npu_port_stat.py \
  --device-id 7 -i 1 -n 30 --counters --csv npu7-ports.csv
```

- 每轮显示各端口 `TxBW(MB/s)`、`RxBW(MB/s)`；`-n 0` 持续采集，Ctrl+C 停止。
- `--counters` 额外输出 `TxCTP/s`、`RxCTP/s`、`TxFlit/s`、`RxFlit/s` 和计数器实际间隔。
  不需要计数器时省略此参数，减少查询开销。
- 用 `--ports 0:4,1:9` 限制为指定的 `Udie:Port`；必须替换成 `--list` 中实际存在的 UB 端口。
- 默认串行；`--workers 4` 可并行查询。每个端口的查询窗口仍有偏移，不能视为严格同时采样。
- `-i` 是每轮完成后的等待时间；计数器速率使用各端口自身的实际时间差。
  `--csv` 写入新文件，已有文件不会覆盖。字段缺失或失败以 N/A/CSV 空值表示，不填 0。

## 指标含义

带宽直接来自以下命令，沿用官方 NPU Exporter 的 `-time 100` 设置：

```bash
hccn_tool -g -bandwidth -i 7 -u 1 -p 9 -time 100
hccn_tool -g -stat -i 7 -u 1 -p 9
```

端口号只是示例。`Bandwidth TX/RX: ... MB/sec` 已经是硬件采样窗口的带宽，
**不再做差，也不除以 Python 的 `-i`**。`query_seconds` 是命令调用耗时，
不是硬件计数器窗口。带宽窗口和 `--counters` 的前后快照区间也不相同。

`--counters` 对累计字段做差：

| 输出 | 来源字段 | 说明 |
| --- | --- | --- |
| TxCTP/s、RxCTP/s | `ub_umoc_ctph_cnt_tx/rx` | CFG7 CLAN 类报文包率，非端口所有报文总包率，非 READ opcode 过滤 |
| TxFlit/s、RxFlit/s | `tx_busi_flit_num`、`rx_busi_flit_num` | 原始计数器单位/秒；不沿用 CPU 脚本的 20 bytes/flit 假设 |

计数器减少时本轮对应速率为 N/A，并用当前值重新建立基线；不猜测位宽或回绕。
若服务器版本不支持其中某个字段，该列为 N/A；若查询失败或完全不能识别输出，
记录错误并返回非零状态。有限 `-n` 在持续失败时仍会结束。

BatchGet 的 READ 请求由 NPU 发往 CPU，数据响应由 CPU 返回 NPU，因此主要关注
**NPU RX 与 CPU TX**。端口还包含控制报文及其他业务流量；这些是端口级读数，
不是单个 Jetty/EID 的带宽。不要把初始化的 BatchPut、warmup 混入 READ 稳态采样。
该测试每轮还生成请求、提交和同步，因此端口的墙钟窗口带宽不等于 kernel 有效载荷带宽。

若 hccn_tool 不支持这些 A5 查询，可使用支持 UB 的 msprof 版本，在同一稳态阶段
独立采集，例如（设备编号同样需核对）：

```bash
msprof --output=./npu7-ub-profile --sys-devices=7 --sys-period=30 \
  --sys-io-profiling=on --sys-io-sampling-freq=10
```

解析结果的 `ub_*.csv` 包含 `Device_id`、`PortId`、`TimeStamp`、
`UBRxPortBandWidth(MB/s)`、`UBTxPortBandWidth(MB/s)`；此路径提供带宽，不提供上述包率。

## 接口依据与验证

- [官方 hccn_tool 调用与解析实现](https://github.com/Ascend/mind-cluster/blob/master/component/ascend-common/devmanager/hccn/hccn_tool.go)：
  `GetNpuDevNetPortInfo`、`GetNPUInterfaceTrafficNpu`、`GetNPUUbStatInfo`。
- [官方 A5 UB 计数器采集器](https://github.com/Ascend/mind-cluster/blob/master/component/npu-exporter/collector/metrics/collector_for_ub.go)。
- [msprof 系统采集](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0012.html)、
  [UB 输出字段](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1163.html)。

本地解析与失败处理验证：

```bash
python3 -m unittest discover -s tests/diagnostics -p 'test_npu_port_stat.py'
```

本地验证使用模拟输出；实际驱动版本的兼容性、端口对应关系和采样开销需在 A5 上验证。
