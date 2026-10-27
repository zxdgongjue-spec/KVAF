# READ request / response 的 SL、VL 确认

此工具采集服务器证据，不把配置映射表当成抓包结果。主程序不修改 QoS、
不重载驱动、不挂载 debugfs、不生成测试流量。使用 `--device-id` 时会调用已有
HCCP 查询程序，编译辅助程序并创建临时查询 context；不会创建 Jetty 或提交 READ。

## 目前代码能确认到哪里

| 环节 | 证据 | 边界 |
| --- | --- | --- |
| NPU 配置 | `direct_hccp_remote_transport.cc` 把 priority 写入 `attr.ub.priority`，传给 `RaCtxQpCreate` | 项目没有实际 HCCP/设备固件内部实现 |
| READ 提交 | `execute_batch_get_aiv.asc` 构造 READ WQE，填写 TP、远端 EID、token、地址、长度；提交到各 Jetty SQ | 构造代码没有按 Jetty priority 单独写入响应 SL/VL |
| 公开 UDMA 出站配置 | `udma_init_jettyc` 对 RM/RC 执行 `ctx->sl = dev->priority_info[cfg->priority].SL` | 需要与服务器实际驱动匹配；Host 驱动不能自动代表 NPU 内嵌驱动 |
| CPU 本地 Jetty | `cpu_runtime.cc` 设置 `jfs.priority = kHostJettyPriority`，当前常量 15 | 不等于自动 READ response 采用 CPU 本地 JFS 的 SL |
| 自动 READ response | CPU 应用没有逐个 READ 提交 SEND 的数据通路 | 响应选级发生在项目以外的驱动/硬件/固件中 |
| 公开 UBASE 配置 | 收集 `resp_vl = req_vl + ctp_resp_vl_offset` 的请求/响应 VL 对 | 仅证明存在配置配对，不证明具体 READ response 复制请求 SL |

不要把响应方向强行写成“CPU 再查一遍 NPU priority 数字”：请求方的 priority
是本地 API 参数。若响应继承请求的服务类别，继承的可能是 SL/事务元数据，
随后使用响应端的通道规则；它未必存在一个独立的 CPU Jetty priority。

**新增的硬件查询依据：**公开 `udma_query_res_jetty` 从硬件 mailbox 查询
Jetty context，再将 `jettyc->sl` 写入结果的 `priority` 字段；公开
`urma_admin` 将其打印为 `pri`。所以 `pri` 可能是硬件 SL，而不是原始 priority。
必须对照服务器匹配版本的 `udma_dfx.c`；不能从名字直接解释。

公开参考（不声称与服务器二进制同版）：

- [UDMA 出站配置](https://github.com/openeuler-mirror/kernel/blob/OLK-6.6/drivers/ub/urma/hw/udma/udma_jetty.c)
- [UDMA 硬件上下文查询](https://github.com/openeuler-mirror/kernel/blob/OLK-6.6/drivers/ub/urma/hw/udma/udma_dfx.c)
- [UBASE 请求/响应 VL 配置](https://github.com/openeuler-mirror/kernel/blob/OLK-6.6/drivers/ub/ubase/ubase_qos_hw.c)
- [UBASE 硬件 SL→VL 读取](https://github.com/openeuler-mirror/kernel/blob/OLK-6.6/drivers/ub/ubase/debugfs/ubase_qos_debugfs.c)
- [UBASE 分 VL 带宽采样](https://github.com/openeuler-mirror/kernel/blob/OLK-6.6/drivers/ub/ubase/debugfs/ubase_debugfs.c)
- [urma_admin 的 pri 字段](https://github.com/openeuler-mirror/umdk/blob/master/src/urma/tools/urma_admin/admin_cmd_show_res.c)

## 第一轮：两端静态采集

在 KVAF 根目录执行，需要 Python 3。输出目录必须尚不存在。使用当前服务实际
打印的 EID；以下值来自已有日志。若 debugfs 需要 root，请在具有相应读取权限的
终端执行。脚本会保留权限错误，不会自动提权或将缺失数据当成 0。

NPU7 所在服务器：先加载你原来的 CANN 环境，并使用匹配的 `build-npu`：

```bash
python3 tests/diagnostics/collect_read_qos.py \
  --role npu --device-id 7 \
  --eid 00000000003f030000100000df00eb00 \
  --build-dir build-npu \
  --output qos-npu7-static
```

CPU DDR 服务所在服务器：

```bash
python3 tests/diagnostics/collect_read_qos.py \
  --role cpu --urma-device udmac0d1e2 \
  --eid 00000000003f020000100000df080b00 \
  --output qos-cpu-static
```

有匹配的内核/驱动/UMDK 源码时，在相应命令追加
`--source-root /实际源码目录`，可重复。只采集与 priority、SL、VL、响应及查询
有关的指定源码文件片段，并记录源文件 SHA256；提供源码本身不证明与运行二进制匹配。
NPU 查询成功输出 HCCP header 路径时，也会采集同目录中相关头文件片段。

输出包含 `report.json`、各命令原始输出、驱动模块信息、HCCP 映射（NPU）、
对应控制器的 QoS 文件及 `vl_bandwidth.csv`。所有失败、超时、截断均有记录。
`qos_sl_vl_map.txt` 中应同时检查 `sw_vl` 与 `hw_vl`，不一致时不要只采用软件缓存值。

**NPU 的特殊情况：**HCCP 使用 HDC 查询的 `udmac0d0e3` 不一定出现在 Host
Linux 的 `/sys/class/ubcore`。脚本必须先验证完整 EID，才会关联本地设备和 debugfs；
绝不按同名设备替代。若报告 `No local debugfs association`，NPU 侧只取得了
HCCP 配置表，实际 SL/VL 仍需该版本的 NPU DFX 接口或设备侧诊断环境。
此时提供报告和 HCCP header 片段，才能选择正确的接口；不能通过 Host 的
`modinfo udma` 或 Host 同名 debugfs 代替。

## 第二轮：查询仍然存活的 Jetty

工作进程结束会销毁 Jetty，所以必须在测试/服务运行期间采集。
NPU 日志中的 `BATCH_GET_JETTY_PRIORITY ... jetty_id=...` 是 NPU 本地 ID；
CPU 日志中的 `CHANNEL_CREATED ... local_jetty=...` 是 CPU 本地 ID。
分别在对应端命令追加，例如：

```text
--jetty-id 123 --jetty-id 124
```

123/124 是占位示例，必须替换为本轮活跃资源 ID。脚本先用
`urma_admin show dev DEVICE jetty ID`；不支持时尝试公开旧接口
`show_res -d DEVICE --resource_type 7 --key ID`。旧版资源枚举可能不同，
保留原始输出并对照安装版本。Jetty 的 resource key 7 与 statistics key 6 不同。
若 NPU 资源仅 HDC 可见，这条本地查询不会被执行，更不会拿 CPU Jetty ID 替代。

## 第三轮：观察两个方向实际使用的 VL

公开 UBASE 的 `perf_stats` 按端口输出 `vl tx_bw(kbps) rx_bw(kbps)`，UB
采样窗口为 100 ms；服务器实际输出为准。此处是带宽读数，不是累计计数器，
不能做前后相减。VL 汇总也不是按 READ opcode 或 Jetty 过滤的抓包。

建议先运行一个 Jetty，将 NPU priority 依次设为 1 和 3，CPU 配置保持不变。
这两个值来自 NPU7 的 CTP 映射候选，不预设谁更优先。
在已有 CPU 服务和两端 EID 配置下，用现有 runner 产生足够长的负载，例如：

```bash
PYTHONUNBUFFERED=1 DEVICE_ID=7 KVAF_BATCH_GET_JETTY_PRIORITIES=1 \
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --npu-eids 00000000003f030000100000df00eb00 \
  --peer-ub-device udmac0d1e2 \
  --aiv-cores 1 --jettys-per-core 1 --token-bytes 656 \
  --request-length 32768 --miss-count 32768 --max-inflight-total 1024 \
  --warmup 20 --iterations 100000 --no-kernel-profile
```

这里保留环境中原有的 `KVAF_CPU_SERVICE_ADDR`；如未设置，请先设为实际服务地址。
等日志出现 `BATCH_GET_PERF_PHASE_BEGIN phase=batch_get` 后，在两端另一个终端
采样，排除初始化、BatchPut 数据准备和 warmup 阶段。该测试有请求生成间隙，
因此目标是观察 VL 分类，不是把采样带宽当成链路线速。

CPU 采样示例（NPU 端用自己的 role/EID，并确保本地诊断确实可见 NPU）：

```bash
python3 tests/diagnostics/collect_read_qos.py \
  --role cpu --urma-device udmac0d1e2 \
  --eid 00000000003f020000100000df080b00 \
  --label p1-only --samples 60 --interval 0.2 \
  --output qos-cpu-p1
```

先采 idle 基线；然后采 p1-only、p3-only，多轮交替，保持大小、目标内存、EID、
端口、访问模式及其他网络负载相同。两端原始输出和 CSV 均带时间戳，检查采样
确实覆盖 READ 稳态；时钟未同步时不能直接按时间戳匹配跨机报文。

| 观测位置 | 重点读数 |
| --- | --- |
| NPU 本地正确设备 TX | READ request 的出站 VL 活动 |
| CPU 正确设备 RX | READ request 到达方向的 VL 活动 |
| CPU 正确设备 TX | READ response 数据出站方向的 VL 活动 |
| NPU 本地正确设备 RX | READ response 到达方向的 VL 活动 |

端口还可能承载 ACK、信用/协议控制流量或其他进程流量，不能把任何一个非零 VL
直接宣布为 READ response。优先结合多轮隔离实验、负载大小变化和硬件 packet trace。

## 哪些结果可以下什么结论

- Jetty context SL 随 NPU priority 变化：确认选级配置已进入上下文，尚非包级证明。
- 隔离负载时，NPU request 方向的 VL 随 priority 变化：支持请求走了不同类别。
- CPU priority 固定，CPU response 方向的主要 VL 随 NPU priority 变化：支持响应
  分类受到请求侧类别影响；还不能唯一证明“直接复制请求 SL”这一具体实现。
- CPU response VL 不变：不能排除不同 SL 映射到了同一 VL，也不能单凭此判定
  `kHostJettyPriority` 是响应分类来源。
- 需要确认响应 **SL 的确切值以及继承规则**：取得可关联同一 READ 的 request /
  response 报文 trace（opcode、两端、事务标识、SL、VL、方向），或匹配该芯片和
  固件版本的明确规则。VL→SL 常为多对一，不能从 VL 反推出唯一 SL。

要区分“CPU Jetty 配置”和“请求类别”，最终可做两个独立变量的交叉实验：
固定 CPU 配置只改变 NPU priority；然后固定 NPU priority，只改变 CPU JFS
的有效 SL。当前 `kHostJettyPriority` 是编译期常量，采集工具不修改它。
CPU 端若选择 13 和 15，已有日志显示二者又映射到同一 VL 8，不能作为区分来源
的有效 VL 对照；必须先确认选用的 CPU 配置实际对应不同类别。

缺少包级证据或匹配的响应选级规则时，报告中的
`response_inherits_request_sl` 始终为 `unverified`。

## 本地验证

```bash
python3 -m unittest discover -s tests/diagnostics -p 'test_collect_read_qos.py'
python3 tests/diagnostics/collect_read_qos.py --help
```

本地测试覆盖 EID/设备错配、嵌套控制器路径、VL 方向及单位解析、缺失与零值的
区分，并验证配置配对和带宽活动不会被自动升级为 SL 继承结论。Linux/CANN
硬件接口必须由服务器实测验证。
