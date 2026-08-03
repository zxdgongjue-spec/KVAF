# A5 官方 URMA 接口能力边界与 KVAF 门禁

## 结论

当前不能把“安装了 `liburma.so.0`”或“头文件中声明了某个函数”解释为
`AIV + SIMT + URMA + H2D` 已具备。“官方接口”不等于“功能完整”；同一个
入口可能存在符号，却对目标 mode 返回 unsupported，也可能返回 SUCCESS 但
实际是 no-op，或在 cleanup 中只打印底层错误而不向上传播。

KVAF 因此把能力拆成六层，每一层都需要独立证据：

1. **符号层**：Device OS 实际 DSO 可加载，且目标入口确实由该 DSO 定义；
2. **返回码层**：入口接受当前 provider、mode、token、EID 和 endpoint；
3. **功能语义层**：SUCCESS 对应真实创建、建链、notify 或销毁，而不是 no-op；
4. **调用域层**：Host Linux userspace API 有明确契约可从 AICPU/Device OS 调用；
5. **生命周期层**：部分创建、并发 init、逆序 cleanup、错误传播和重试所有权完整；
6. **AIV 私有队列 ABI 层**：队列地址处于 AIV 可访问域，且 WQE/CQE、
   doorbell、owner bit、barrier 和 cache 语义与 A5 一致。

当前代码中的运行状态固定为：

```text
HARDWARE_EXECUTION_DISABLED
ZERO_WQE_NOT_PROVEN
DYNAMIC_RUNTIME_PROBE_DISABLED
```

source-linked zero-WQE 和三种官方 runtime 动态 probe 都不再作为下一步。
前者无法在资源已 armed 后保证失败一定可回收；后者尚不能产生封存的
capability observation ledger，会消耗一次 reboot 窗口却不能机器化解锁下一层。
当前只允许读取文件的 registered-package audit 和 HCCP/HAL 声明 preflight。

## 为什么 HCCP 能工作不能证明 Direct-UMDK 能工作

HCCP 提供的是“已知可工作的集成架构基线”：Host 进程创建 URMA 资源，再通过驱动 HAL 把资源映射到
NPU 进程。资源所有者、Linux fd、ioctl、sysfs 和 pthread 状态都留在 Host
Linux 用户态。Direct-UMDK 当前尝试让 AICPU 直接调用公共 `urma_*`。这改变了
调用者所在的操作系统和进程模型；两条路径不是同一个契约。

“已知可工作”不等于“生命周期实现完美”。已审计的 HCCP 路径自身也存在部分
unmap/unimport 返回值只记录或未向上完整传播的情况。因此 KVAF 借鉴的是它的
Host owner + HAL 映射边界，不会照搬其错误处理，也不会把 HCCP 能运行作为
Direct-AICPU cleanup 已安全的证据。

HCCP 自身还使用了公开安装头没有声明的私有 payload：`UdmaVaInfo` 位于 HCOMM
内部 `rs_ctx_inner.h`，通过公开 `res_map_info_in.priv` 传给
`halResAddrMapV2/halResAddrUnmapV2`。KVAF 因而把它保存为带 HCOMM commit 和三个
源码哈希的 `hccp_hal_private_abi.h`，并对 24 字节大小和 0/4/8/16 字段偏移做编译期
检查。这只证明开源 HCCP 快照的布局，**不**把它冒充公开 HAL API，也不证明服务器上
`libascend_hal.so` 与该快照运行时配套。公开符号面、源码审计私有布局和运行时配套是
三个独立门禁。

当前 UMDK 公共实现具有明显的 Host Linux 依赖：`dlopen` provider、扫描
sysfs、打开 `/dev/uburma`、执行 ioctl，并在对象中保存 fd 和 pthread 锁。
在已审计的公开源码和文档中尚未找到公共 `urma_*` 可由 AICPU/Device OS 调用
的明文保证；这不等价于断言厂商内部一定没有该能力。KVAF 因此分别记录
`documented_official_aicpu_call_contract=not-found-in-audited-sources` 与
`local_runtime_compatibility=unproven`。只有获得配套 SDK/文档，或在具备可封存
观察账本和恢复契约的条件下完成本机行为验证，后者才可以变为 `measured`。

## 已确认的官方接口不完整或实现风险

以下结论针对当前固定的 UMDK/libummu 源码快照，不外推到所有未来版本。
Device OS 实际加载的 `liburma.so.0` 尚无内容哈希/源码版本握手，因此历史探针设计会把
“已审计固定源码存在缺口”和“实际 runtime 来源未知”作为两个字段输出，不能把前者
直接当成后者的版本证明，也不能因实际库符号更多就反推缺口已经修复。
换言之，下表能证明 pinned UMDK 快照不可作为“官方实现天然完备”的
oracle，却不能证明服务器 Device OS 正在运行的 DSO 必然具有同一缺陷。
实际 DSO 的构建来源、provider 依赖闭包和底层服务对象在获得可验证的版本
身份前一律记为 `unproven`。

| 风险 | 当前证据 | 对 KVAF 的影响 |
|---|---|---|
| API 版本不能代表目标能力 | 固定源码声明 `URMA_API_VERSION=0.9`，但 alloc/set/get/active/deactive/free 等入口仍受具体 provider、mode 和 token 限制 | 版本号和编译头不能代替 Device DSO 身份、具体能力和语义证据 |
| AICPU 调用域无已发现的公开契约 | 公共 core 依赖 Linux dl/sysfs/device/ioctl/pthread；在已审计公开源码/文档中未找到 Device OS/AICPU caller-domain 契约 | init/create 即使成功也只能算本机实验结果 |
| `urma_alloc_jfs` 失败路径不安全 | provider 返回后未先检查 status/`*jfs`，就增加引用并解引用对象 | 暂不调用现代 alloc 行为；先只探测符号 |
| `urma_alloc_jfr` 失败路径不安全 | 与 JFS 相同，失败返回可能被继续解引用 | 暂不调用现代 alloc 行为 |
| Jetty advise/unadvise 尚有 TODO | public header 标注等待 RM/IB_RC 实现 | 不能假定所有 transport mode 的建链完整 |
| 部分公开函数可能是“成功空操作” | 固定源码中 UB `urma_advise_jetty`/`urma_unadvise_jetty` 可直接返回 SUCCESS；`urma_ack_notify` 校验 ops 后也可不调用 provider 而返回 SUCCESS | 符号存在甚至 SUCCESS 都不能独立证明功能语义 |
| 公共 API 可表达范围不等于 UDMA provider 支持范围 | 公共 cfg 要求 mode 受设备支持；当前 UDMA JFS 实现明确拒绝 RC，JFR 仅接受 NONE/PLAIN_TEXT，拒绝更高 token policy | capability 必须按具体 device/mode/token 做行为验证 |
| libummu 部分销毁不返回底层错误 | 某些 munmap/PLBI/超时结果被忽略或接口为 void | 不能仅凭 public return 证明资源已释放；KVAF 当前策略要求 checked teardown，无法证明时保留 quarantine |
| `urma_uninit` 不传播全部 provider close 失败 | 固定源码的 `urma_close_provider()` 只记录 provider `dlclose` 错误，外层 `urma_uninit()` 仍可返回 SUCCESS | public uninit + 外层 dlclose 成功不能证明内部 provider 已卸载 |
| process-global init 排他性不足 | 固定源码先读取全局 init flag，完成 provider/sysfs 初始化后才增加；不是 compare-exchange，也没有覆盖整个初始化的公开全局锁 | probe 自身 mutex 只能保护同一 probe DSO，不能证明 AICPU 共享进程无并发 caller |
| 已审计公共队列地址契约仍不完整 | public header 暴露通用 VA/PI/CI/DB-status，但未规定这些 VA 的 AIV 可见地址域及 WQEBB/CQE、owner/opcode/token、barrier/cache ABI | 不能仅凭地址非零让 AIV 写 SQ/读 CQ |
| HCCP 映射 payload 未公开 | 安装 HAL 只公开 `res_map_info_in.priv`；实际 `UdmaVaInfo` 布局位于 HCOMM 内部源码 | 可以固定源码审计布局做候选兼容层，但运行时配套未证明前不得调用 HAL 映射 |

## 历史设计：运行时探测的结果语义（当前禁用）

> 下文记录的 `discover`、`init-device` 和 `create-delete` 是已实现但未获准
> 执行的分层 probe 协议，不是当前操作命令。
> `run_a5_aicpu_runtime_urma_probe.sh` 会在 CANN、NPU 和官方 DSO 访问之前
> 固定输出 `DYNAMIC_RUNTIME_PROBE_DISABLED` 并退出 2。只有未来实现了可封存、
> 可校验、可在下一 boot 消费的 observation ledger，且不确定 cleanup 不会
> 被误判为安全时，才能重新评审这些模式。

原设计中，`discover` 只在一个 AICPU 调用中执行
`dlopen`/`dlsym`/`dlclose`，不调用
`urma_init`，不主动创建 URMA/UMMU 对象。但它并非“绝对无副作用”：`dlopen` 会执行
官方 `liburma` constructor；当前源码中的 constructor 会初始化日志并发现 sysfs
路径。因此 discover 也必须由 detached supervisor 持锁并先发布 quarantine，不能用
timeout 强杀；`dlclose` 只能证明本次 loader handle 已释放，不能证明 constructor 的
全部进程/内核副作用已逆转。`dlopen`/`urma_init` 返回失败也可能发生在部分初始化之后，
这些情况必须保留隔离状态，不能按“安全失败”清除。探针使用 v2 nonce、请求字段回显
和逐字段范围校验，避免旧 DSO 或损坏结果被误判。KVAF 自有 Device probe DSO 采用
每次构建生成独立的 256-bit build-instance ID 并用于 SONAME，同时另记源文件、编译器、
构建脚本、内核 UAPI 及 UMDK/libummu revision 派生的 source/config 摘要；Host 会验证
期望 AICPU package 内 ELF 哈希，Device 则回显完整 build-instance ID。这能证明 loader
请求的 versioned SONAME 与 probe 回显的构建
身份一致，但不能证明 Device loader 实际映射 ELF 的逐字节哈希；状态字段因此明确叫
`expected_packaged_provider_artifact_sha256`。它也不能证明 probe 随后加载的官方
`liburma.so.0` 来自哪份源码，后者继续报告为 `runtime_liburma_source_identity=unproven`。
输出中的符号面分组为：

按原协议，`discover` 会是一次性的、按 boot 消耗的诊断：即使符号解析与 `dlclose` 都成功，
结果也保持 `cleanup_proven=0`，AICPU owner 与 quarantine 保留到 OS 重启或厂商提供等价
恢复证据。日志中的 symbol surface 只能供离线判断，不能形成 safe terminal，也不能直接
进入下一阶段。若不能接受一次重启窗口，应只运行 Host 侧 HCCP/HAL 声明与符号预检，
不要启动 AICPU `discover`。

- `legacy_lifecycle_symbol_surface`：旧 create/delete context/JFC/JFR；
- `candidate_modern_queue_wrapper_symbol_surface`：未被当前 KVAF 路径调用的
  alloc/set/get/active/deactive/free 候选接口；
- `segment_symbol_surface`：register/import 及反向操作；
- `candidate_public_posting_symbol_surface`：JFS/JFR post 与 poll 的候选公开数据面；
- `legacy_jetty_link_wrapper_symbol_surface`：旧 create/import/bind wrapper；
- `current_*_symbol_surface`：从当前 checked Direct-UB 实际代码路径审计出的 endpoint、
  resource、CTP link、management I/O 和 teardown 五组；
- `current_direct_ub_symbol_surface`：当前实际路径所需 34 个公开符号的精确并集。

即使最后一项是 `complete-semantics-unproven`，也只说明这些入口与解析到的
`urma_init` 来自同一个 DSO。它不证明 dlopen handle 的完整依赖关系；导出符号仍可能
是 stub、拒绝当前 mode/token，或带有已知生命周期
缺陷。以下输出仍应保持 `unproven`：官方函数语义、AICPU caller domain、AIV 地址
可见性、WQE/CQE 私有布局、doorbell、内存屏障，以及官方 `liburma.so.0` 的源码身份。

`init-device` 第一次调用公共 URMA 生命周期，只执行 init、按名称查设备、uninit。
即使这些公开 API 都返回成功，当前也只记录 `official_public_cleanup=returned-complete`；
由于官方没有 AICPU caller-domain/constructor 内部副作用契约，不能由此证明进程与内核
状态已恢复，所以仍保持 owner/quarantine，等待重启或厂商等价恢复证据。
此外，`urma_init()==SUCCESS` 本身不证明至少一个目标 provider/device 可用；固定实现即使
未留下可用设备也可能完成 init，因此必须继续检查按名称 discovery，且仍将并发全局
初始化排他性报告为 `process_global_exclusivity=unproven`。
三个 AICPU 模式都必须由显式风险接受开关启动，并由独立 supervisor 记录进程身份、
boot id、日志哈希和最终状态。任何不确定退出都不得自动删除状态或继续 Zero-WQE。

`create-delete` 仍是后续 A/B 实验，不是生产路径。它在同一次 AICPU invocation
中创建 context/JFC/JFR 并逆序销毁，以避免把进程内 handle 跨 KFC launch 保存。
只有对 `discover` 日志完成离线审阅、随后重启清除其不确定调用域，并且
`init-device` 的证据符合预期后才允许安排该实验。

## 决策表

下表是未来动态 probe 重新获准后的离线分类规则，不是当前的运行顺序。
当前决策只有一条：先执行 registered AICPU package 静态审计和 HCCP/HAL
声明 preflight；不论两者是否通过，都不解锁 zero-WQE、动态 probe 或 H2D。

| 结果 | 根因分类 | 下一步 |
|---|---|---|
| Device `liburma.so.0` 不可加载 | Device OS 未提供对应运行库/搜索路径 | 停止 Direct-UMDK；检查匹配的 Device 包 |
| 基础符号缺失 | 运行库过旧或被裁剪 | 停止；不能进入 init |
| 基础存在、modern/Jetty/Segment 不完整 | 官方功能面不足以支持目标数据面 | 评估 HCCP/HAL 控制面或匹配新版 Device 包 |
| 符号齐全、init 被拒绝 | AICPU 调用域不支持、endpoint 不匹配或运行时 ABI 不兼容 | 保留原始 status/errno；不创建队列 |
| init 成功且公开逆序 API 返回成功 | 仅证明本次公开调用序列返回成功；官方内部副作用仍无契约 | 保留隔离并重启后离线评估，不直接串行下一阶段 |
| lifecycle 的公开返回值全部成功 | 仍未证明内部 cleanup、AIV 可见性和私有队列 ABI | 保留隔离；下一 boot 才能安排 inactive queue 哨兵实验 |
| 任一 cleanup 不确定 | 资源所有权不可证明 | 保持 owner/quarantine，走厂商恢复或重启 |

同一 boot 内不会用操作者的风险接受文本替代 cleanup 证据。若探针进入官方 DSO 后只能
证明自身公开 handle 已关闭，而不能证明 constructor/runtime 内部副作用已逆转，状态会
保持 blocked/quarantine。只有 OS 重启后 boot ID 变化，runner 才允许执行一次
recovery-only barrier 清理：有 terminal 的 blocked 会核对 terminal/log/hash；因
`HoldUnsafeOwner` 而没有 terminal 的会核对 quarantine/active/ready 的全部会话、worker、
provider 和路径身份，同时明确不把未封存日志当作 capability 证据。该次命令不会顺带
启动新的 probe。

## 最小证伪实验

当前的最小证伪是纯静态的：

1. `audit_a5_registered_aicpu_packages.sh` 检查 CANN registry 已注册 archive 中的
   SONAME、成员名和 ELF hash 是否冲突。冲突非零时，当前 package/runtime
   身份基线被证伪，应先修复部署，不进入硬件；
2. `preflight_a5_hccp_hal_capabilities.sh` 只编译安装 HAL header 并读取
   `libascend_hal.so` 动态符号表。声明面缺失时，Host HCCP/HAL 候选控制面
   在该安装上被证伪；声明面完整时，也只能记为
   `declarations-complete/runtime-unproven`，不能记为 backend ready。

两项静态证据收集后，再决定是否转向 HCCP/HAL Host 长驻控制面，或向厂商
索要官方 AICPU caller-domain、生命周期和 AIV 队列 ABI 契约。在任一选项形成
新证据前，继续排列 JFR flag、重跑 zero-WQE 或以重启换取下一次动态探测
都不能提高根因判别力。
