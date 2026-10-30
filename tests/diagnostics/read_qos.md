# CPU 侧 URMA 信息与 port/VL 带宽采集

两个入口均在 CPU 服务所在的 Linux 服务器运行：

| 入口 | 使用时机 | 输出 |
| --- | --- | --- |
| `collect_read_qos_info.py` | 首次使用，或设备、内核、驱动、UMDK、QoS 配置变化后 | `report.json`、CPU 环境/驱动/配置原始文件、可选源码片段 |
| `collect_read_qos.py` | 每轮测试期间，可同时采集多个设备 | 每个设备子目录中的 `perf_*.txt` 和 `sampling.log`，以及总目录的运行日志 |

两个脚本放在同一目录，采样脚本复用信息脚本中的设备校验和命令执行函数。
采样不要求预先生成信息快照，也不读取旧报告中的设备路径。
两者均不修改 QoS、不重载驱动、不挂载 debugfs、不生成测试流量。

已移除 `--role`、`--device-id`、`--build-dir`、`--hccp-include`、
`--jetty-id`。旧 CPU 命令也需要删掉 `--role cpu`。
这两个入口不再执行 NPU/HCCP/Ascend 查询、辅助程序编译或 Jetty/TP 资源查询。

## 运行前准备

在 KVAF 根目录执行，需要 Python 3 和 CPU 服务器上的 `urma_admin`。
将 `QOS_CPU_EID` 设置为本轮 CPU 服务实际使用的本地 EID，
`QOS_CPU_DEVICE` 设置为经完整 EID 核对的本地 URMA 设备名。
`--urma-device` 可省略，此时每个 EID 必须唯一匹配一个本地设备。
信息脚本仍一次获取一个设备的信息；采样脚本支持多个设备/EID。

输出目录必须尚不存在。若 debugfs 需要 root，请在有相应读取权限的终端执行；
脚本不会自动提权，也不会把权限错误或缺失读数当成零带宽。
换服务器或设备后，需要重新确认 EID 和设备名。

## 获取 CPU 信息

```bash
python3 tests/diagnostics/collect_read_qos_info.py \
  --urma-device "${QOS_CPU_DEVICE:?先设置本轮 CPU 设备名}" \
  --eid "${QOS_CPU_EID:?先设置本轮 CPU EID}" \
  --output qos-cpu-info
```

采集内容如下：

| 文件 | 内容 |
| --- | --- |
| `os_release.txt` | CPU 服务器的操作系统版本 |
| `urma_version.txt`、`urma_help.txt`、`urma_devices.txt` | URMA 工具信息、本地设备/EID 信息 |
| `modinfo_<module>.txt`、`loaded_<module>_srcversion.txt` | `ubase`、`udma`、`ubcore`、`uburma` 四个模块的元信息和已加载构建标识 |
| `caps_info.txt`、`all_adev_name.txt` | 对应 CPU 侧 UBASE 控制器的能力及设备信息 |
| `qos_*.txt` | `adev_qos`、`sl_vl_map`、`ets_tc`、`ets_tcg`、`ets_port`、`tm_queue`、`tm_qset`、`tm_pri`、`tm_pg`、`tm_port` 原始输出 |
| `report.json` | CPU 环境、设备关联、配置解析、查询状态和命令起止时间 |

信息脚本不读取 `perf_stats`，不生成采样 CSV；报告也不包含采样数量、采样状态、
逐次采样记录或预设的请求/响应 SL 判断。配置中的请求/响应 VL 配对仍会从
`adev_qos` 的 `ctp_req_vl` 与 `ctp_resp_vl_offset` 提取。

有匹配的 CPU 侧内核、驱动或 UMDK 源码时，可追加
`--source-root /实际源码目录`，支持重复指定。仅扫描指定的 CPU 侧驱动/UMDK 文件，
保存与 priority、SL、VL、响应及统计相关的源码片段，并记录源文件 SHA256；
不扫描 HCCP 头文件。提供源码本身不证明与运行中的驱动二进制匹配。

环境与配置未变时可以复用这些文件，不需要随消息大小、迭代次数等普通测试参数
重复生成。QoS/TM 文件仍是当时的配置/状态快照，不是永久不变的硬件常量。
`qos_sl_vl_map.txt` 中应同时检查 `sw_vl` 与 `hw_vl`，不一致时不能只采用软件缓存值。

设备/EID 或 debugfs 关联失败时，脚本打印 `INFO_UNAVAILABLE`，保存已有报告并以
状态码 1 退出。`local_device_eids` 给出解析到的设备/EID，`local_device_error`
记录关联失败的阶段和原因。其他信息命令的失败、超时、截断记录在 `commands` 中。

## 采集 CPU port/VL 带宽

在测试负载进入需要观察的阶段后，在 CPU 服务器的另一个终端执行：

```bash
python3 tests/diagnostics/collect_read_qos.py \
  --urma-device "${QOS_CPU_DEVICE:?先设置本轮 CPU 设备名}" \
  --eid "${QOS_CPU_EID:?先设置本轮 CPU EID}" \
  --label test-A --samples 60 --interval 0.2 \
  --output qos-cpu-test-A
```

同时采集多个设备时，一次传入多个设备名和 EID，两组值按顺序一一对应：

```bash
python3 tests/diagnostics/collect_read_qos.py \
  --urma-device "${QOS_CPU_DEVICE_A:?先设置设备 A}" "${QOS_CPU_DEVICE_B:?先设置设备 B}" \
  --eid "${QOS_CPU_EID_A:?先设置设备 A 的 EID}" "${QOS_CPU_EID_B:?先设置设备 B 的 EID}" \
  --label test-A --samples 60 --interval 0.2 \
  --output qos-cpu-test-A
```

也可以重复写 `--urma-device DEVICE_A --eid EID_A --urma-device DEVICE_B --eid EID_B`。
设备名与 EID 的数量必须一致，每个设备只能列一次；省略设备名时可用
`--eid EID_A EID_B` 自动匹配。所有设备共用本轮的 label、samples、interval 和 timeout。

每次运行只查询一次当前设备列表，逐个校验 EID → 本地设备 → UBASE debugfs
控制器的关联，然后并发读取不同控制器的 `perf_stats`。设备校验在内存中完成，
不另存 `urma_devices.txt`。多个设备若关联同一个控制器，每轮只读取一次，
将相同原始数据和读取时间写入各设备目录，并在设备日志标记 `SHARED_CONTROLLER`。
这些副本仍是同一控制器的汇总读数，不能当作各设备独立的流量再相加。

按实际设备名创建子目录，例如：

```text
qos-cpu-test-A/
  sampling.log
  udmac0d1e2/
    perf_0000.txt
    perf_0001.txt
    ...
    sampling.log
  udmac0d0e3/
    perf_0000.txt
    perf_0001.txt
    ...
    sampling.log
```

单设备命令的参数写法仍然适用，但结果也统一放入 `--output/<设备名>/`，
不再直接写到总输出目录。

- 总目录的 `sampling.log`：设备发现、校验失败、控制器分组及整体完成状态。
- 设备目录的 `perf_*.txt`：每次读取的原始输出，保留端口及 VL 的 TX/RX 信息。
- 设备目录的 `sampling.log`：CPU EID、设备/路径、label、主机/内核、采样参数、每次读取的
  起止 Unix 时间、状态和最终完成数量，不重复保存端口/VL 带宽值。

`--samples` 默认为每个控制器 1 次，每个关联设备保存同样数量的样本；
`--interval` 默认为 0.2 秒，最小 0.1 秒，表示同一个控制器两次读取之间的
等待时间，不包含读取本身耗时；`--timeout` 默认为每条命令 15 秒。
不同控制器的读取并发执行，不保证硬件采样窗口严格同步，应以日志时间为准。
`--label` 只是本轮测试标签，不设置或筛选任何优先级。

采样脚本不生成 `report.json`、`vl_bandwidth.csv` 或驱动配置文件。
用日志中的 `name=perf_XXXX` 将原始文件与起止时间对应；原始文件本身不添加时间戳。

设备关联失败时打印 `SAMPLING_BLOCKED`，原因记录在总日志；通过校验的其他设备
继续采样。读取失败、超时、截断或无法识别 `vl tx_bw(kbps) rx_bw(kbps)` 表时打印
`SAMPLING_FAILED`，停止该控制器的后续采样，不影响其他控制器。
已有原始输出和错误日志会保留。任一设备未完成时整体返回状态码 1，全部完成时返回 0。
`valid_perf_samples=0` 表示没有有效样本，不代表带宽为 0。

## 读数范围

`perf_stats` 中的 TX/RX 方向均相对于所选 CPU 侧控制器。每个样本可以包含
多个 port，每个 port 可以包含多个 VL；脚本保留全部原始输出，不按端口或 VL 过滤。
带宽单位为 kbps，是窗口带宽读数而非累计计数器，不能对相邻读数做差来计算带宽。

这些读数是端口/VL 上的汇总流量，不按进程、EID、READ opcode 或 Jetty 过滤。
可以先采 idle 基线，再在负载稳定阶段采样。CPU RX/TX 中可能混有其他负载和控制
流量，仅凭某个 VL 非零不能确定具体报文类别，也不能从 VL 反推出唯一的 SL。

## 本地验证

```bash
python3 -B -m unittest discover -s tests/diagnostics -p 'test_collect_read_qos.py'
python3 -B tests/diagnostics/collect_read_qos_info.py --help
python3 -B tests/diagnostics/collect_read_qos.py --help
```

测试覆盖 CPU 设备/EID 校验、控制器路径、port/VL 方向和零值、两个入口的输出边界、
已移除参数的拒绝、源码范围、多控制器并发、共享控制器去重、按设备分目录、
单个设备失败后的继续采样，以及读取失败和超时后的部分结果。
测试临时目录会在结束时清理。实际 Linux URMA/UBASE 接口需在 CPU 服务器验证。
