# A+K 首次 BatchPut 本地访问错误诊断

用于 `cq_status=2 cq_sub_status=2` 等首次写入错误的对照采集，不需要旧环境版本。
这一状态按仓内参考 UDMA provider 映射为本地访问错误，仍应核对实机驱动版本。
测试不预设配置错误：`non_pin` 对照仅用于检验环境变化后的注册/映射兼容性，
重点是 kernel 返回的 CQ 错误、请求 metadata 与内存注册证据。默认运行行为不变；诊断选项只有
`KVAF_DIAGNOSTICS=1` 时生效。仅在没有其他 KVAF 工作负载的测试时段运行。

## 运行

CPU 端继续使用已有服务和参数，本轮先回传 NPU 端终端报告即可。
本分支没有修改 CPU 协议，不要求升级 CPU 服务。

NPU 端先按正常方法加载 CANN 环境，再重新构建；不能使用旧 binding/host library：

```bash
./build.sh npu
KVAF_CPU_SERVICE_ADDR=141.61.80.71:29876 DEVICE_ID=0 \
  bash tests/run_remote_cpu_diagnostics.sh "$PWD/build-npu"
```

脚本直接在终端打印 `DIAG_REPORT_BEGIN` 到 `DIAG_REPORT_END` 的诊断报告，
**本次诊断脚本的总终端输出最多 1000 行**（构建命令输出不包含在内）。直接复制粘贴终端输出即可，
不需要上传文件夹，也不需要先收集 CPU 服务日志。报告包含每组关键原始证据、异常尾部、
实际加载库、构建指纹和新出现的内核错误。原始完整日志仍在本地保留作备查。任何 case 失败时
脚本最终退出码为 1，这是诊断结果，不表示日志采集一定失败。`dmesg` 不可读时会记录原因。
目录包括 summary.json、manifest.json（构建文件 SHA256）、environment.txt、每组独立日志、
运行前后的 dmesg。case 内记录实际加载的相关动态库路径，不输出完整进程环境。

默认四组，均为独立进程，同一个 CPU/NPU EID 候选组合：

| case | core | Jetty/core | Put HBM non_pin |
| --- | --- | --- | --- |
| c1-j1-nonpin1 | 1 | 1 | 1（原行为） |
| c1-j1-nonpin0 | 1 | 1 | 0（唯一注册模式变量） |
| c32-j2-nonpin1 | 32 | 2 | 1 |
| c32-j2-nonpin0 | 32 | 2 | 0 |

每组只分配 256 KiB CPU DDR，先运行原有初始化写探测，再用 torch HBM 写入不同内容，
最后 BatchGet 回读并逐字节校验。初始化探测使用 aclrtMalloc 源内存，后续写入使用 torch 源内存。
readback 的 Get 注册保持原行为；因此 Put 成功后 Get 失败需要单独解释。

默认固定 `peer_candidate=0`、`npu_candidate=0`，与提供的失败日志一致。初始化不会自动换候选，
以免把 EID 差异混入 non_pin 对照。若需要测试另一组合：

```bash
KVAF_CPU_SERVICE_ADDR=141.61.80.71:29876 DEVICE_ID=0 \
  bash tests/run_remote_cpu_diagnostics.sh "$PWD/build-npu" \
  --peer-candidate 0 --npu-candidate 1 --single-only --timeout 120
```

序号是本次枚举顺序，必须以日志中的实际 EID 核对跨进程一致性。
可以用 `--output /path/to/new-directory` 指定未存在的输出目录；
`--dry-run` 仅显示矩阵，不接触硬件。

## 判读

- `DIAG_PHASE_PASS phase=initialize_probe`：原初始化探测成功。
- `phase=torch_hbm_put`：torch 源 HBM 写入阶段。
- `phase=readback`：回读与正确性阶段；不能把它的失败算作 Put 失败。
- `HBM_REGISTER`：实际 flags/non_pin、输入及注册地址范围、token 分配结果和注册返回结果。
  `allocated_wqe_token` 已右移 8 位；`returned_raw_token` 原样记录，并另列移位结果，不能直接比较未统一编码的数据。
- `RA_LIBRARY` / `DIAG_LOADED_LIBRARY`：实际加载库；`HOST_ABI`：本地手写接口结构大小。
- `ENTITY`：从 device 回读的 core 0 每条 lane 的地址、token、TPN、队列进度。
- `PUT_COMPLETION`：直接打印 kernel 已写入自有 HBM status 的 CQ status/substatus，先于附加回读输出。
- `QUEUE_ADDRESSES`：只记录队列地址，不用主机 ACL 读取 SQ/CQ。旧诊断在该操作前后出现 SIGSEGV，
  不能把 RA 返回的队列 VA 当作普通 HBM 地址。当前仅回读 KVAF 自己申请的 HBM 描述符、metadata 和状态。
- `HBM_REGISTER implementation=batch_put_private_v2`：确认执行了 Put 独立的注册实现。
  原 Get/Put 动态库有同名全局辅助函数，可能使 Put 注册调用绑定到 Get 实现；本版本将 Put 辅助函数私有化。
- `acl!=0`：自有 HBM 回读失败，不能把空 words 当有效内容。
- `DIAG_CLEANUP_OK`：客户端 teardown 返回成功，不证明所有底层 destroy 已成功或 CPU 服务无残留；
  应结合 CPU `REGION_RELEASE`。超时、清理异常或缺少诊断编译标记时停止剩余矩阵。
  超时强制结束进程可能留下 CPU session，确认 CPU 服务资源恢复后再重跑。

若 non_pin=0 通过而 1 失败，优先定位注册/映射模式；两者都在同一 `(2,2)` 失败，则结合
device entity 的本地 token/VA、请求 metadata、注册返回值及 UMMU fault 继续定位。单队列通过而 32×2 失败，才优先查规模和隔离。

此分支不改 kernel WQE 编码或 token 使用方式。诊断会增加同步完成后的回读开销，不用于性能测量。

诊断 v2 必须重新运行 `./build.sh npu`。上一版若以 exit=-11 退出且没有 REGION_RELEASE，
应先确认 CPU 服务残留 session 已清理；不能把该段错误当原始 CQ 故障的根因。
