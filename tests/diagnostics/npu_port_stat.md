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
python3 -u tests/diagnostics/npu_port_stat.py \
  --device-id 7 --workers 8 -i 0.2 -n 0 \
  > npu7-ports.log 2>&1
```

- 每轮显示各端口 `TxBW(MB/s)`、`RxBW(MB/s)`；`-n 0` 持续采集，Ctrl+C 停止。
- `--counters` 额外输出 `TxCTP/s`、`RxCTP/s`、`TxFlit/s`、`RxFlit/s` 和计数器实际间隔。
  会先读取一次计数器基线，后续每端口每轮也多一次 `-stat` 查询；仅看带宽时省略。
- 用 `--ports 0:4,1:9` 限制为指定的 `Udie:Port`；必须替换成 `--list` 中实际存在的 UB 端口。
- 默认 `--workers 8`，并行采集端口，整轮结束后一次打印完整表格并立即 flush。
  每个端口的查询窗口仍有偏移，不能视为严格同时采样。
- `-i` 是两轮开始之间的目标间隔，包含本轮查询和打印耗时；耗时超过目标间隔时，
  下一轮立即开始，不额外等待，也不并发叠加两轮查询。计数器速率仍使用各端口自身的实际时间差。
- 只输出到终端，不生成 CSV。上面的 `2>&1` 同时保存诊断错误；若希望边看边保存，
  将重定向改成 `2>&1 | tee npu7-ports.log`。字段缺失或失败显示 N/A，不填 0。

## 查询耗时与可调参数

旧版默认 `--workers 1`，每个端口查完立即打印，而且整轮完成后再等待 `-i` 秒。
`stat.py` 则默认使用 8 个线程，等所有结果返回后统一打印。新版已采用同样的并行与整轮输出方式，
并将等待改为补足目标周期。`hccn_tool -bandwidth` 与 `ubctl pkt_stats` 是不同查询，
实际查询耗时仍由服务器工具/驱动决定，不能保证设置 `-i 0.2` 就达到每秒 5 轮。

每轮表头的 `collect` 是整轮查询耗时，`start_dt` 是相邻两轮开始的实际间隔
（开启计数器时第一轮与基线比较），`target` 是设置的目标间隔。
每行 `Query(s)` 是该端口本轮所有命令的总耗时，不包含等待线程调度的时间。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--device-id` | 必填 | hccn_tool 的 NPU logic ID |
| `--workers` | `8` | 并行采集端口数；可尝试 `16` 并比较 `collect` 是否下降 |
| `-i / --interval` | `1` | 两轮开始的目标间隔，单位秒，支持如 `0.2` 的小数 |
| `-n / --count` | `1` | 采样轮数，`0` 为持续采集 |
| `--ports` | 全部 UP 的 UB 端口 | 指定 `Udie:Port` 列表，缩小查询范围，如 `0:4,1:9` |
| `--counters` | 关闭 | 额外采集计数器包率/flit速率，会增加查询量 |
| `--timeout` | `10` | 每条命令的超时秒数；只限制失败等待，不会加快正常查询 |
| `--list` | 关闭 | 只列出 UB 端口 |
| `--hccn-tool` | PATH 或驱动默认路径 | 指定 hccn_tool 可执行文件 |

想缩短周期，先省略 `--counters`，再增加 `--workers` 或用 `--ports` 缩小范围。
若工具或驱动内部将查询串行化，增加线程数可能无法进一步缩短周期；根据日志中的实际耗时判断。

## 指标含义

带宽直接来自以下命令，沿用官方 NPU Exporter 的 `-time 100` 设置：

```bash
hccn_tool -g -bandwidth -i 7 -u 1 -p 9 -time 100
hccn_tool -g -stat -i 7 -u 1 -p 9
```

端口号只是示例。`Bandwidth TX/RX: ... MB/sec` 已经是硬件采样窗口的带宽，
**不再做差，也不除以 Python 的 `-i`**。`Query(s)` 是命令调用耗时，
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

## 接口依据与验证

- [官方 hccn_tool 调用与解析实现](https://github.com/Ascend/mind-cluster/blob/master/component/ascend-common/devmanager/hccn/hccn_tool.go)：
  `GetNpuDevNetPortInfo`、`GetNPUInterfaceTrafficNpu`、`GetNPUUbStatInfo`。
- [官方 A5 UB 计数器采集器](https://github.com/Ascend/mind-cluster/blob/master/component/npu-exporter/collector/metrics/collector_for_ub.go)。

本地解析与失败处理验证：

```bash
python3 -m unittest discover -s tests/diagnostics -p 'test_npu_port_stat.py'
```

本地验证使用模拟输出；实际驱动版本的兼容性、端口对应关系和采样开销需在 A5 上验证。
