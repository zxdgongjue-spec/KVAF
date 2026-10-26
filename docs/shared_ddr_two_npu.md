# 两张 NPU 访问同一块 CPU DDR

本实验验证：NPU A 通过 AIV BatchPut 写 CPU DDR，NPU B 通过自己的链路执行
AIV BatchGet，能否读到完全一致的数据。默认再交换两张卡的角色运行一次。

## 构建和运行

CPU 与 NPU 必须重新构建同一分支。本功能将控制协议从 v6 升级到 v7，不能混用旧服务。

CPU 机器：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

NPU 机器需有两张可用卡、torch_npu、NumPy、CANN，以及到 CPU 的 TCP 和 UB 连通性：

```bash
./build.sh npu
KVAF_CPU_SERVICE_ADDR=<cpu-ip>:28999 \
  ./tests/run_remote_cpu_shared_ddr.sh "$PWD/build-npu" \
  --devices 0 1 --blocks 8 --rounds 100
```

快速单向测试：

```bash
KVAF_CPU_SERVICE_ADDR=<cpu-ip>:28999 \
  ./tests/run_remote_cpu_shared_ddr.sh "$PWD/build-npu" \
  --devices 0 1 --blocks 1 --rounds 3 --one-way
```

默认 `token_bytes=656`、`tokens_per_block=128`，每轮读完整个 block。
可以使用 `--token-bytes`、`--tokens-per-block`、`--cores`、`--jettys` 调整。
`--timeout` 是每个协调阶段的等待秒数，默认 180。

A 会选择一个可用的 CPU candidate。B 必须能够访问 A 选中的同一个 CPU UB device/EID，
不能自动新建另一个 region。可在 CPU 启动时通过 `--ub-device NAME --eid-index N`
固定两卡都可达的候选；也可在测试脚本传 `--peer-ub-device NAME` 限定设备，具体 EID
由 A 选择后固定给 B。两卡的本端 EID 各自发现。

## API 与控制协议

创建者保持现有 API：

```python
writer.link_to(cpu_service)
writer.initialize(ddr_bytes, slot_size)
region = dict(writer.remote_region)
```

通过测试协调进程将 region 字典传给另一个进程，读卡使用：

```python
reader.link_to(cpu_service)
reader.initialize_shared(region)
```

`remote_region` 包含 `server_instance_id`、`region_id`、`candidate_id`、`session_id`、
`base`、`length`、`slot_size`、`device`。挂接只使用身份和布局字段，不信任调用方提供的
裸 base，也不复用创建者的 session_id。CPU 返回自己记录的注册描述和一个新 session_id。
服务实例 ID 在每次启动时随机生成，region ID 在该实例内单调分配，避免旧 handle 因地址
重用而挂到新分配。该 handle 是资源标识，不是鉴权凭据；实验沿用已有受控环境的服务模型。

- `AllocateRegion`：分配并注册一次 DDR，同时建立创建者会话。
- `AttachRegion`：校验服务实例、region、大小、slot size、candidate，创建独立会话。
- `CreateChannel`：绑定到请求的会话，两个会话各自拥有 Jetty/JFC/JFR。
- `ReleaseRegion`：在 v7 中解除当前会话；仅在最后一个会话退出时注销 segment/token 并释放 DDR。
- 挂接不存在或不匹配的 region 直接失败，绝不回退到独立分配。

普通初始化保持 AIV WRITE/CQ 启动探测。共享挂接执行 **只读** BatchGet/CQ 探测，
读第一段数据以选择可用本端 EID，不写 DDR、不创建 Key，也不运行带宽选优；
`adaptive_ub_device=True` 与共享挂接不兼容。探测 buffer 保持到 transport 关闭后才释放。
只读探测验证连通性和完成状态，payload 正确性由正式实验判断。

## 测试顺序与判定

使用 multiprocessing spawn 启动一张卡一个进程。父进程仅转发控制消息、region 描述和
地址表，不传输 payload。

1. A 初始化，B 挂接。核对两端 instance/region/base/length/slot/candidate 一致，session 不同。
2. A 根据轮次和 block 序号生成伪随机数据，BatchPut 同步成功后导出地址/长度/有效位。
3. B 接收 WRITE_DONE，独立生成期望值，在本卡创建地址表。所有 token 按打乱的顺序读取，
   输出行反向映射；输出预填为期望值逐字节取反，漏读不会被旧值掩盖。
4. B 等待 BatchGet stream 完成，逐字节比较，成功后返回 READ_OK。
5. A 收到确认才释放 AccessTable、BatchFree，再向相同 slot 写下一轮。检查每轮地址不变。
6. 最后一轮 A shutdown，B 再读同一批地址，验证创建者退出后 DDR 仍有效。
7. B shutdown 后再次挂接旧 handle，必须得到 NotFound，确认最终释放。
8. 默认交换两张卡并用新的 region 重复以上测试；`--one-way` 可关闭。

B 使用现有接受 `addresses`/`valid` Tensor 的 BatchGet 接口，不调用自己空 MetaManager
上的 Key 查询。跨 Runtime 没有共享 Key 锁或 slot 分配器，实验由单写者和读完成确认保护
slot；这些 API 不提供并发多写者的元数据一致性。

成功输出包含 `ROUND_OK`、`DIRECTION_OK ... post_detach_read=pass stale_attach=rejected`，
最后为 `PASS: shared DDR two-NPU correctness`。CPU 日志中每个正式 region 应只有一次
`REGION_ALLOCATED`，随后有 `REGION_ATTACHED` 和不同 session 的 `CHANNEL_CREATED`；
自动候选探测可能另有创建后释放的临时 region，最终以脚本打印的 region ID 为准。

错误输出包含轮次、region、读卡、远端地址、首个错误字节和总错误字节数；CQ 错误由已有
AIV 日志输出。不通过重复读取直到一致来掩盖错误。正确性测试包含同步和逐字节校验，
输出不作为吞吐性能结果。

阶段超时会使实验失败，协调器先要求进程退出并 detach；硬件调用卡死时会终止子进程。
当前服务尚无异常退出租约回收，强制终止后应重启 CPU 服务回收遗留会话。

## 控制面回归测试

CPU 构建会生成 `kvaf_shared_ddr_control_test`。它使用真实 CPU 服务与 PeerLink 协议，
通过本地 TCP 和模拟 UMDK provider 验证共享注册、独立会话、失败回滚、旧身份拒绝及最终释放，
运行时不需要 UB 设备：

```bash
ctest --test-dir build-cpu -R kvaf_shared_ddr_control_test --output-on-failure
```

该测试不验证硬件缓存可见性、实际 RMA 数据传输或驱动完成语义；必须另行运行双卡实验。
