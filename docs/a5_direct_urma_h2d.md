# A5 AIV + SIMT + direct URMA + H2D

## 1. 结论

KVAF 现在有一条不调用、不链接 HCOMM 的实验性 H2D 路径：

```text
Host DDR(HVA)
  -> halHostRegister(HOST_MEM_MAP_DEV)
  -> Host DVA
  -> Host liburma 注册 MR、创建 Jetty/TP
  -> TCP 交换 EID、MR、Jetty、TP
  -> AICPU liburma 注册 HBM MR、创建 Jetty/SQ/CQ
  -> 返回 AICPU shared device-VA 的 SQ/CQ/doorbell/TP/MR
  -> 32-lane SIMT 规划 descriptor
  -> 单 AIV 直写 64B UDMA READ SQE
  -> SQ doorbell
  -> DCCI 后轮询 CQE，并更新 CQ doorbell
  -> Host DDR 数据进入 Device HBM
```

这里复刻的是 `UB_MicroBenchmark_new` 已跑通的资源模型：普通
`urma_create_jetty` 在 AICPU KFC 进程内执行，AIV 消费同一 unified
device-VA。代码不会把这个事实误标成 `CREATE_JETTY_EX/TGID-VA`；
最终可达性由真实的一条 SQE、一个 CQE、payload 和 canary 联合验证。

直接依赖中已经移除：

- memcache；
- memfabric；
- HCOMM headers、`libhcomm.so` 和 HCOMM resource API。

仍然需要：

- CANN/ACL/runtime、bisheng、Ascend HAL 和驱动；
- AICPU KFC 动态加载；
- 系统 `liburma.so`、`liburma_common.so`、`liburma-udma.so` 和内核 UDMA；
- 与运行时 UDMA 同代的 `urma_perftest/ubengine` 源码快照。

这里的“绕过 HCOMM”是指 KVAF 自己完成资源控制。EID、MR、Jetty、TP
仍必须创建和交换，只是不再由 HCOMM/memfabric/memcache 代办。

## 2. 代码来源与职责

### 2.1 借鉴 `urma_perftest`

复用并在 KVAF provider 中源码编译的部分包括：

- `perftest_parse_args` 和配置校验；
- TCP server/client 与两端配置交换；
- `urma_create_context`；
- JFC、Jetty、token、segment 的创建参数和顺序；
- EID、segment、Jetty、TP 信息交换；
- remote segment/Jetty import 和 connect；
- 对称 teardown 的基本顺序。

KVAF 不再链接预编译 `liburma_perftest.so`。构建脚本把选中的源码、
Host/Device provider、系统 URMA 动态库哈希和 build-id 绑定到 manifest，
避免旧 SO 被误打包。

上游 fork 的 `fill_ai_jetty()` 会按一份复制的 private struct 读取 queue。
KVAF 的 checked 路径不调用它，而是仅用 public URMA API 创建资源，完成后
由一个独立 accessor 按实测 ABI 只读提取 SQ/CQ，避免创建阶段混用两套
private layout。

### 2.2 借鉴 `UB_MicroBenchmark_new`

设备数据面复刻其 AIV 直写 UDMA 的关键规则：

- 64B READ SQE；
- remote EID 的硬件字节序；
- 24-bit TP ID；
- public `token_id >> 8` 得到 20-bit UDMA TID；
- READ local SGE 的 token word 保持 0，使用已注册 local VA；
- SQ doorbell offset；
- CQ owner/phase、DCCI 和 CQ consumer doorbell。

KVAF 增加了 descriptor 越界检查、单 owner、stream 同步、结果/canary
验证和失败隔离。

### 2.3 Host DDR

Host 端严格区分两个地址：

- HVA：CPU 填充和校验 DDR；
- DVA：`halHostRegister(..., HOST_MEM_MAP_DEV, ...)` 返回，交给 URMA 注册。

生命周期必须满足：

```text
分配 DDR
  -> Host 注册得到 DVA
  -> URMA register/import
  -> 数据传输和 CQ 完成
  -> 两端 disconnect/unimport
  -> 两端确认
  -> unregister MR/token
  -> 删除 Jetty/JFC/context
  -> Host unregister DVA
  -> 释放 DDR
```

在不能证明 DMA 已停止时，代码不会主动释放 backing memory，而是写入
quarantine。

## 3. 为什么要分成 Host/Device 两个 provider

标准 `urma_query_*` API 不返回 userspace SQ/CQ 的 `qbuf`、硬件
doorbell 或 JFC `sw_db`。`urma_user_ctl(CREATE_JETTY_EX)` 只能解决新建
SQ 的部分信息，也不能查询普通 JFC 的完整 CQ，因此当前 AIV direct
路径仍需要一个受控的 UDMA 私有访问层。

两端使用同一个经过 `UB_MicroBenchmark_new` 随附 SO 验证的
`urma/code/include/urma_types.h` AIV private layout，但编译成两个角色
不同的 provider：

| 端 | provider private ABI | KVAF 产物 |
|---|---|---|
| Host | `PERFTEST_AIV_V1 / Host` | `libkvaf_urma_provider_host.so` |
| AICPU/Device | `PERFTEST_AIV_V1 / AICPU shared VA` | `libkvaf_urma_provider_device.so` |

不能用 `ubengine/ssapi` 里另一代 private header 替代它：两代 public
JFC/Jetty base 和 queue tail 不同，会让 SQ、CQ、`sw_db` 全部错位。
`CREATE_JETTY_EX(tg_id)` 适用于其他进程代建并把 doorbell 映射到目标
TGID 的路径；本实现的资源本来就在 AICPU 内创建，强行改用它反而偏离
已验证基线。

KVAF 把 private header 限制在
`provider/private_udma_accessor.c` 一个翻译单元中，并为 Host/Device 角色
分别编译。
它从真实 queue 对象导出实际 `qbuf_size/baseblk_cnt`，并在返回任何地址前
校验：

- AArch64 public JFC/Jetty、private queue/context 的固定大小和关键偏移；
- SQ/CQ ring 非空、2 的幂、64B stride；
- qbuf 边界和 baseblock 数一致；
- PI/CI 初始值为 0；
- SQ 非 reserved 模式；
- CQ 软件 doorbell 非空；
- provider config 所声明的 Host/Device ABI 与 SO 一致。

构建阶段还会实际 `dlopen` 两个 SO，调用
`kvaf_urma_provider_private_abi_v2()` 证明角色，不能只靠 manifest 文本
声称 Host/Device 身份。

运行脚本仍拒绝 `sq_reserved=1`，因为该模式的设备地址语义还没有在 A5
上验证。

## 4. 资源创建与字段来源

两端 provider 执行：

```text
parse/check config
  -> establish TCP connection
  -> create context
  -> create JFC/Jetty
  -> register local segment
  -> exchange EID/MR/Jetty/TP
  -> import peer segment/Jetty
  -> connect
  -> validate exact private queue ABI
  -> provider-ready 双端确认
```

AIV 使用的字段来源如下：

| 字段 | 来源 |
|---|---|
| SQ base/depth/doorbell | `PERFTEST_AIV_V1` Device private accessor |
| CQ base/depth/sw_db | `PERFTEST_AIV_V1` Device private accessor |
| remote MR | `remote_seg[0].ubva/len` |
| remote TID | remote public segment `token_id >> 8` |
| local object ID | local public segment `token_id >> 8`，仅用于资源校验，不写入 compact READ WQE |
| remote EID | imported target Jetty public EID，AIV 只反转一次 |
| TP ID | imported target Jetty 的 `tp.tpn` |
| token policy | `URMA_TOKEN_NONE`，SQE `tokenEn=0` |

不能使用旧 sidecar 的 `ai_jettys[].sq.tp_id`。当前 UDMA provider 没有实现
`urma_ops.get_tpn`，该值可能是负错误码转换后的无符号数。

## 5. 失败与 teardown

创建过程按阶段互相确认。一端失败会 shutdown TCP 控制连接，另一端不会
无限阻塞。正常关闭顺序是：

```text
AIV stream 完成
  -> SQ PI == CQ CI
  -> 两端 disconnect/unimport Jetty
  -> barrier
  -> 两端 unimport peer segment
  -> barrier
  -> unregister local segment/token
  -> 删除 Jetty/JFC/context
```

runner 只有在两个进程退出码、数据/canary、provider close marker 全部精确
通过时才打印：

```text
KVAF_A5_AIV_SIMT_DIRECT_URMA_H2D_ACCEPTANCE=PASS
```

超时、stream 同步失败、provider cleanup 不确定或 marker 缺失会生成：

```text
/run/lock/kvaf-a5-direct-urma.quarantine
```

此时先按站点流程 reset/硬件隔离所选 NPU，再终止记录的 PID 和删除
quarantine。不要在 reset 前 `kill -9`。

## 6. 操作命令

### 6.1 服务器 B 宿主机：更新代码

执行环境：`SERVER_B_HOST`

```bash
echo "EXECUTION_ENV=SERVER_B_HOST"
cd /data/c00982601/KVAF

git switch A5_AIV_URMA_cs
git pull --ff-only origin A5_AIV_URMA_cs
git log -1 --oneline

test -r \
  /data/c00982601/urma_perftest/ubengine/ssapi/userspace/udma/udma_u_common.h

docker ps --filter name=vllm-ascend-cs-kvaf
docker exec -it -u 0 vllm-ascend-cs-kvaf bash
```

不需要新建容器。`/data` 已由宿主机挂载到同路径，后续命令都在
`vllm-ascend-cs-kvaf` 内执行。

### 6.2 Docker：加载环境

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
echo "EXECUTION_ENV=VLLM_ASCEND_CS_KVAF_DOCKER"
cd /data/c00982601/KVAF
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
export KVAF_URMA_PERFTEST_ROOT=/data/c00982601/urma_perftest
export KVAF_DIRECT_PROVIDER_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma-provider
export KVAF_REFERENCE_PERFTEST_LIBRARY=/data/c00982601/UB_MicroBenchmark_new/3rd/urma_perftest/liburma_perftest.so
```

若 `ldconfig` 中同一个 SONAME 有多份候选，显式指定实际运行库：

```bash
export KVAF_URMA_LIBRARY=/usr/lib64/liburma.so.0
export KVAF_URMA_COMMON_LIBRARY=/usr/lib64/liburma_common.so.0
export KVAF_UDMA_PROVIDER_LIBRARY=/usr/lib64/urma/liburma-udma.so
```

`KVAF_REFERENCE_PERFTEST_LIBRARY` 只用于把
`UB_MicroBenchmark_new` 已验证过的 AIV private layout 产物记录进
build manifest，KVAF provider 不链接它。若该路径不存在，构建脚本才回退到
`urma_perftest/urma/code/output` 中的同名产物。

### 6.3 Docker：构建两端 provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/build_a5_direct_urma_provider.sh

ls -l \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_host.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_device.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.manifest" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.success"
```

脚本执行 fresh build，校验导出符号、AArch64 ELF、`DT_NEEDED`、RPATH、
源码前后哈希、Host/Device private-ABI probe，以及实际 URMA/UDMA runtime
哈希。

### 6.4 Docker：安装 AICPU provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/package_a5_direct_urma_aicpu.sh

tar -tzf \
  "$ASCEND_HOME_PATH/compat/liburma_perftest.tar.gz"
```

archive 名称保留 `liburma_perftest.tar.gz` 只是为了适配当前 CANN compat
loader；内容不再包含旧 `liburma_perftest.so`，精确白名单是：

```text
aicpu_kernels_device/libkvaf_urma_provider_device.so
aicpu_kernels_device/kvaf-direct-urma-provider.manifest
aicpu_kernels_device/kvaf-direct-urma-source.sha256
```

### 6.5 Docker：构建 AIV/SIMT smoke

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
if test -r \
  /data/c00982601/asc-devkit-9.1.0-20260722/include/simt_api/asc_simt.h
then
  export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit-9.1.0-20260722
else
  export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit
fi

export KVAF_A5_DIRECT_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma
bash scripts/build_a5_direct_urma.sh
```

该 target 显式设置：

```text
KVAF_A5_BUILD_HCOMM_BACKEND=OFF
KVAF_A5_BUILD_DIRECT_URMA_BACKEND=ON
```

所以 direct binary 和 provider 的 `DT_NEEDED` 都不能含 `libhcomm`。

### 6.6 Docker：确认 UDMA device/EID index

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
urma_admin show --all --whole | less
```

然后填写实际值，不要照抄示例：

```bash
export KVAF_HOST_URMA_DEVICE=<Host侧UDMA设备名>
export KVAF_HOST_EID_INDEX=<Host侧EID序号>
export KVAF_DEVICE_URMA_DEVICE=<Device侧UDMA设备名>
export KVAF_DEVICE_EID_INDEX=<Device侧EID序号>
```

Host/Device peer EID 默认由
`scripts/resolve_a5_urma_eids.py` 从 `/proc/asdrv_ub/pair_info` 解析，不依赖
memcache。

### 6.7 Docker：运行 H2D

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
unset KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
export KVAF_H2D_BYTES=4096
export KVAF_DIRECT_CONTROL_PORT=21115

bash scripts/run_a5_direct_urma_h2d_smoke.sh
```

若存在多个 active pair，按错误信息指定：

```bash
export KVAF_PAIR_CHANNEL=<实际channel>
bash scripts/run_a5_direct_urma_h2d_smoke.sh
```

成功必须同时看到 Host DDR 注册、两端 provider ready、direct channel、
数据/canary、两端 teardown 和最终 acceptance marker。证据保存在
`results/a5_direct_urma_h2d_<run-tag>/`，其中还包含 provider build/source
manifest。

## 7. 是否必须升级 CANN

绕过 HCOMM 本身不要求升级 CANN，也不再受
`COMM_PROTOCOL_UBC_CTP/ChannelHandle` 这类 HCOMM header 版本限制。

当前版本能否使用，以四项实测为准：

1. bisheng 能编译 `dav-3510` AIV + SIMT；
2. ACL/runtime 能执行 custom AICPU KFC；
3. compat loader 能加载上面的 Device provider；
4. 系统 `liburma`/`liburma-udma` 与提供的 `ubengine` 源码同代。

四项均通过就不必仅为 direct URMA 升级 CANN；任一项缺失才需要补匹配组件
或升级完整套件，不能只替换某个 include 目录。

## 8. 仍未完成的生产化工作

当前实现用于 A5 正向 MVP 和性能验证，还不能宣称生产安全：

- private accessor 仍依赖 `PERFTEST_AIV_V1` 布局。最终应在实际
  `liburma-udma` provider 中新增 versioned
  `QUERY_DIRECT_QUEUE_INFO`，由 provider 自己返回 SQ/CQ 地址和 feature，
  彻底删除外部 private cast；
- 当前 UDMA 源码中的 `udma_u_unregister_seg`/`udma_u_free_tid` 会吞掉
  `ummu_ungrant`/`ummu_free_tid` 的底层失败并返回成功。要可靠释放
  Host DDR/HBM，必须修改并部署匹配的 `liburma-udma.so`，让错误向上传递
  且失败句柄保持可重试；KVAF 的 quarantine 只能降低风险，不能修复它；
- provider 是进程级 singleton，仅支持一个连接；
- AICPU Open/Close 必须命中同一进程和 SO 实例；
- Host DVA 的 `non_pin` 注册和 AIV CQ cache 语义仍需当前 A5 固件实测；
- 当前逐 descriptor `doorbell -> CQ`，还没有批量 WQE/doorbell/CQ。

正确性通过后的性能工作依次是：长驻 context、Host DDR pool 一次注册、
SIMT 批量生成 WQE、一次 doorbell 提交一批、尾部 fence/CQE、多 SQ 分片。
绕过 HCOMM 主要减少控制面依赖与建链开销；稳态带宽仍取决于长驻资源和批量
提交。
