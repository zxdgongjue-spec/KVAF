# KVAF Host CPU RoCE 读写方案

## 目标与验证基础

为当前 KVAF Runtime 增加可显式选择的 `roce` 后端。A5 Host CPU 构造并提交 RoCE READ/WRITE、轮询完成队列。RoCE 网卡在 NPU HBM 和远端 CPU DDR 之间搬运 KV 数据。两端通过 TCP 交换 DDR 和 QP 参数。

[Host RoCE demo](../access_operator/demo/host_roce/README.md) 已在跨机环境通过 64 B WRITE、READ 和回读校验。客户端使用 `hrn5_4`、GID `192.168.1.101`，服务端使用 `hrn5_0`、GID `192.168.1.90`。完整路径已验证 HAL 登记、HBM MR、RC QP 和两笔 CQE。应用 Tensor 的实际分配方式、多请求批次，以及 READ 后 NPU 算子的可见性仍需在 Runtime 接入时单独验收。

首版采用单 session、单 RC QP、单 CQ。业务选择 `transport="roce"` 并配置 RoCE 端点后，继续使用现有的 BatchPut、BatchGet 和 AccessTable 入口，沿用原参数、NPU Tensor 布局、返回值和 stream 语义。每批先查 HBM 注册，对可追踪生命周期的 buffer 复用 MR，其余 buffer 按批注册和注销。RoCE 路径使用 Host verbs、CANN Runtime 和 Ascend HAL，无需 NDA 或设备侧通信算子。

## 对外兼容约定

| 现有业务调用 | RoCE 首版要求 |
| --- | --- |
| `batch_put(keys, source, input_offsets, lengths, stream)` | 保留源 NPU Tensor、参数和同步返回行为。成功返回时 Key 已提交。 |
| `batch_get(device_access_table, indices, offsets, miss_count, output_indices, output, ...)` | 保留 NPU Tensor 参数和 `DeviceAccessTable` 入口。使用关联 Handle 持有 slot pin，按 stream 顺序读取设备表。 |
| `batch_get(addresses, valid, indices, offsets, miss_count, output_indices, output, ...)` | 保留裸 NPU 地址表重载。Host 按 stream 顺序读取设备表的当前值，并检查读取范围属于当前 RoCE 会话的 DDR region。 |
| `stream=None`、`synchronize=False`（Get 默认） | 沿用当前 NPU stream，提交后可返回。同一 stream 的下游任务在本批 READ 完成且结果可见后执行。显式 `synchronize=True` 等待完成并报告结果。 |

首版沿用当前业务的 NPU Tensor 请求参数。CPU Tensor 入口留作可选扩展。仓库已有 [ACL Graph BatchGet 测试](../tests/python/test_npu_batch_get.py)。如果现有业务在图内调用 Get，原图直接切换 RoCE 并可重复回放是该业务的上线条件。图内兼容按下文的真机验证判定。

## 模块接入

| 模块 | 实现改动 |
| --- | --- |
| [NpuRuntime](../runtime/src/npu_runtime.cc)、[后端配置](../runtime/include/npu_runtime_config.h) | 增加 `Transport::kRoce` 与内部 RoCE Host 后端。在 `LinkTo()` 建立会话对象，在 `CreateCore()` 注入 RoCE allocator 与 operator。在 `Initialize()` 中按 RoCE 配置独立初始化，直接使用指定网口，与 URMA 的 EID 候选流程分开。允许 RoCE 构造时省略 AIV kernel 路径。 |
| [RuntimeCore](../runtime/src/runtime.cc)、[AccessOperator](../access_operator/include/access_operator.h) | 复用远端 DDR 分配接口、Put 事务、AccessTable 和 Key 生命周期。沿用当前 NPU 参数和地址表校验。为有 Handle 的异步 Get 保留内部操作引用，将 slot 解锁延迟到仍引用它的操作实际完成。 |
| RoCE 连接与传输 | 将 [demo 的 verbs 和 TCP v7 握手](../access_operator/demo/aicpu_roce/roce_connection.cc) 抽成共享 Linux 库。RoCE session 持有控制 socket、PD、QP、CQ、协商能力和远端 MR 描述。allocator 申请和释放远端 region，RoCE operator 执行 HBM 注册、请求校验、提交和完成。两个 demo 与 Runtime 链接同一份连接实现。 |
| 远端 CPU 服务 | 从 [固定 64 B DDR 的 demo 服务](../access_operator/demo/aicpu_roce/cpu_service.cc) 提取会话处理。生产入口使用独立监听端口，允许多个有上限的会话。每条长连接保持 DDR、MR 和 QP 存活，接受同一 v7 的申请、建链和释放命令。可复用 RuntimeService 的通用监听代码，会话由独立 worker 处理，避免长连接占住 accept 循环。 |
| [Python 绑定](../bindings/src/npu_python_runtime.cc)、构建 | 增加 `transport="roce"`、`rdma_device`、`rdma_port`、`gid_index`。两个现有 Get 重载进入同一 RoCE 后端，沿用默认参数。扩展根目录、Runtime 和绑定的 CMake 条件，支持仅构建 RoCE CPU 服务或 RoCE NPU 后端。 |

配置中的 `device_id` 是 ACL 逻辑设备 ID，两端的 RDMA device、port、GID index 分别配置。Host CPU 用 ACL 设备 ID 选择 HBM 所在 NPU，用 RDMA 配置打开关联 RNIC。RoCE 首版沿用现有 `batch_get_token_bytes` 和 `batch_get_tokens_per_block` 的读取布局约定。

历史提交 `a28de08` 已有一版 Runtime 集成和测试，可用 `git show a28de08 --stat` 审查。当前实现以已通过真机的 demo 和当前 `main` 的调用约束为基准，按模块提取历史代码。

## 初始化与资源所有权

1. 客户端用指定 RNIC 打开 verbs context，创建 PD、CQ、RC QP，进入 INIT。检查本地端口和 GID，记录实际 GID、QPN、MTU 与 READ credits。
2. NPU Runtime 的 RoCE allocator 通过 TCP 请求 `ddr_bytes` 和 `slot_size`。远端服务检查申请大小、配额、对齐及设备 MR 能力，分配并注册 DDR，配置远端 READ/WRITE 权限，返回 base、length、rkey。
3. 两端交换 QPN、PSN、GID、MTU 和 READ credits，按协商值推进 RTR、RTS。operator 初始化成功后，`RuntimeCore` 再登记 region 的 slot 元数据。64 B WRITE/READ 可作为部署冒烟，并应在业务 Key 可访问前运行。
4. 客户端以一条长连接管理一个 region 和一个 QP。远端 DDR 地址通过现有 `RemoteRegionDesc.base` 进入元数据层，rkey 留在 RoCE session；所有读写目标范围需落在该 session 的远端 MR 内。断开后旧 region 和 AccessTable 失效，重新初始化后重新发布业务数据。

## BatchPut

1. `RuntimeCore::BatchPut()` 验证输入和偏移长度，调用 `PreparePut()` 预留 slot，得到每条 Key 的远端 DDR 地址。
2. Host 等待调用方 stream 的生产工作完成。operator 为源 HBM 查找可复用的 verbs MR，未命中时注册。
3. 每笔 WRITE 使用 `input.data + input_offsets[i]`、`lengths[i]`、本地 lkey 和远端目标地址/rkey。提交前校验所有范围和长度。
4. Host 按窗提交 WR，检查本批全部 CQE。结束本批对 MR 的引用后返回，Core 再调用 `CommitPut()` 发布 Key。提交或完成异常时，须先确认停止 DMA，再让 Core `AbortPut()` 释放预留 slot。

## BatchGet

1. 在调用方 stream 上排入元数据 D2H，拷回 `miss_count`、`indices`、`offsets`、`output_indices` 以及设备表的 `addresses/valid`，使用有界的预分配 Host 缓冲区。先按容量拷贝，Host 只解释 `miss_count` 指定的有效前缀。设备表的当前内容是本批地址来源，这样能保持上游更新设备表后的原有行为。有 Handle 时可以额外使用 Host 表的长度和 pin。
2. Host 校验 `miss_count`、索引、valid、offset、表项长度、输出行、地址溢出和 MR 边界。裸设备表沿用配置的表项长度，并验证最终访问落在当前 region 内。零条请求走正常完成路径，无需 RDMA 提交。
3. 为目标 HBM 查找具备本地写权限的 verbs MR，未命中时注册。每笔 READ 的远端地址为 `addresses[indices[i]] + offsets[i]`，本地地址为 `output.data + output_indices[i] * output_block_bytes`。发生目标范围重叠时，沿用业务约定的结果语义，不引入新的入口参数约束。
4. Host 提交 READ、等待本批 CQE，并完成平台要求的设备可见性操作。将完成状态接回调用方 stream，再允许下游读取 HBM。有效 Handle、HBM 注册和相关缓冲区在实际完成前保持存活。部分完成后失败时，整批输出无效。

### 保持默认异步调用

候选方案是在元数据 D2H 后插入有序的 Host 任务。调用线程先准备本批状态、Host 缓冲区和 MR 引用，再把 D2H 与任务排入现有 stream。等 D2H 完成后，Host 任务通知 RDMA worker。worker 解析请求、发 READ 并查 CQ。确认数据可见后，完成门控再放行同一 stream 的下游任务。默认 Get 在提交成功后返回，显式同步等待同一任务的完成。

[`aclrtLaunchHostFunc()`](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_1959.html) 是图外 stream 门控的候选。CANN 对 [Host 回调](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/others/acldevg/runtime_doc_dev_0015.html) 有调用约束，回调只协调 CPU 状态，ACL 资源操作和需要的设备可见性步骤安排在回调之外。RDMA worker 发布完成前，目标 stream 正在等待它，因此 worker 此时不能同步等待这条 stream。否则 stream 等 worker、worker 又等 stream，会造成循环等待。平台是否还要求额外的缓存或可见性操作，需要根据目标驱动接口和下游 NPU 算子实测确认。还需验证 RDMA 异步失败能通过现有同步或图执行边界暴露，并且失败后下游不会把部分写入的 HBM 当成成功结果。

### ACL Graph 验证

在目标 CANN 9.2.1/A5 上做最小原型。用原来的入口 capture，再多轮回放且每轮换请求参数。验证 D2H 读取的是本轮数据、Host RDMA 任务每轮执行一次、下游在完成后继续，以及 graph reset 能安全回收所持资源。普通 stream 回调文档只能作为图外原型依据，图内触发和等待机制以真机验证为准。图内路径验证通过前，RoCE 可作为开发验证后端，图内业务仍使用已支持该语义的路径。

## HBM 注册与长期复用

和 AIV URMA 一样，每次 Put/Get 先查注册，未命中才新建。RoCE 的缓存保存 **HBM 注册范围、MR 句柄和本地 lkey**。Host 每批根据本批偏移、长度和远端地址生成 WR。

1. **定位本地分配。** Put 查询源 HBM，Get 查询目标 HBM。核对所属 NPU，用 `aclrtMemGetAddressRange()` 获取实际分配的基址和大小，并校验切片和实际读写范围在该分配内。请求的偏移和长度用于计算地址和检查覆盖范围。
2. **查找能覆盖请求的 MR。** 以可验证的 HBM 分配身份、NPU 设备、当前 PD 及 MR 地址范围和权限做匹配。只要已有 MR 覆盖本批各笔本地访问且权限足够，不同偏移和长度就能使用同一个 lkey。首版在网卡 MR 容量允许时，按整个可复用分配建立 MR。READ 目标需要本地写权限，使用带本地写权限的 MR 也可以作为 WRITE 源。
3. **未命中时注册。** 对实际分配先申请或复用进程内的 HAL 登记，再在当前 PD 下调用 `ibv_reg_mr()`。同一 HBM 分配在多个 PD 下使用时，各 PD 持有各自的 MR，并共用同一 HAL 登记。注册失败时回收本次新取得的资源。
4. **持有引用到传输结束。** 缓存只接受由内存池/调用方持有、能明确通知退役的分配，或由 KVAF 持有能保证 backing 存活的引用。只有地址和长度的裸指针，以及无法确认分配生命周期的 buffer，按批临时注册并在本批结束时解除。即使新分配落在同一地址，也要使用新的分配身份。
5. **有界回收。** 缓存设置条目数和占用字节数上限，超限时回收空闲条目；没有可回收条目时可以走临时注册。buffer 退役或 session 关闭时停止获取新的引用，等待在途操作完成或确认停止 DMA，再注销对应 MR。最后一个引用释放后解除 HAL 登记。临时注册与长期缓存共用 HAL 引用管理。

## 完成、失败和关闭

- 首版为每条 WR 请求完成项，以 SQ、CQ 容量限制在途数，READ 同时受协商 credits 限制。每窗提交后轮询，校验 WR ID、操作和 WC 状态。当 `ibv_post_send()` 部分成功或 CQ 异常时，记录已提交与已完成数量。
- 遇到部分提交、错误 CQE 或超时，先使 session 失效并停止 QP。只有 provider 已确认 DMA 不会继续访问相关地址，才允许向 Core 返回普通失败、归还 slot 和 HBM。现有 Core 收到 Put 错误会立即执行 `AbortPut()`。因此首版若销毁 QP 失败、且 KVAF 无法持有调用方 HBM，采用进程 fail-stop。要支持进程内恢复，须一起改造 Core 的 slot 隔离和 buffer 所有权。
- 正常关闭先停止接收新的任务，等待已排入 stream 的工作和 RDMA worker 完成，再停止本地 QP，注销本地 HBM 注册并发释放控制消息。等待期间保持回调、元数据缓冲区和 MR 引用有效。服务端先停止目标 QP，再注销 DDR MR 并释放 DDR，最后回 ACK。TCP 断开走同一服务端清理顺序。清理失败时隔离对应资源并保留诊断，直到安全回收。

## 实施顺序与验收

| 阶段 | 通过条件 |
| --- | --- |
| 1. 抽取共享连接与 CPU 会话 | 两个 demo 的原有构建入口可用。CPU 服务支持配置的 DDR region、多客户端配额和断连清理。用已知跨机网口重跑 64 B 冒烟。 |
| 2. 接入 Runtime 和 BatchPut | 通过 `transport="roce"` 使用真实 NPU Tensor。覆盖变长块、非连续偏移、多个 WR 窗口、Key 提交可见性和失败后 slot 处理。验证同一 HBM 分配的不同偏移能复用 MR。 |
| 3. 接入 BatchGet | 原业务用例仅切换传输配置。覆盖两个 NPU Tensor 重载、默认异步与显式同步、设备表动态更新、输出重排和零请求。用同一 stream 上游生成描述、下游消费结果，验证顺序、错误和可见性。图内业务另以原图多轮回放通过为上线条件。 |
| 4. 回归与测量 | 覆盖 PD 变化、地址重用、缓存容量、buffer 退役、注册失败、部分提交、CQ 超时和断连。进行现有后端回归，分别测 stream 等待、描述 D2H、MR 首次注册/缓存命中、提交与 CQ 等待。 |

## 构建和验证

远端 CPU 机器只需 Linux、verbs 开发库和可用的 RoCE 网口。

```bash
cmake -S . -B build-roce-cpu -DKVAF_BUILD_ROCE_CPU_SERVICE=ON -DBUILD_TESTING=OFF
cmake --build build-roce-cpu --target kvaf_roce_cpu_service -j
build-roce-cpu/runtime/kvaf_roce_cpu_service \
  --listen 90.96.15.111 --port 29999 \
  --rdma-device hrn5_0 --rdma-port 1 --gid-index 1
```

服务端默认允许 64 个会话、单 region 4 GiB、累计 16 GiB。用 `--max-sessions`、`--max-region-bytes` 和 `--max-total-bytes` 调整。监听地址用于 TCP 握手，RoCE 数据面的地址由各端的 RDMA GID 决定。

A5 客户端加载目标 CANN 环境后，构建 Python 绑定并运行现有 Runtime 集成用例。以下参数对应已通过 demo 的两台机器。

```bash
./build.sh npu --with-roce
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_RUN_NPU_TESTS=1 KVAF_CPU_SERVICE_ADDR=90.96.15.111:29999
export DEVICE_ID=0 KVAF_ROCE_DEVICE=hrn5_4 KVAF_ROCE_PORT=1 KVAF_ROCE_GID_INDEX=1
python tests/python/test_npu_batch_get.py --transport roce
```

两个后端在同一进程中的配置和验证命令见[本地 GatherRead 与远端 RoCE](../runtime/README.md#本地-gatherread-与远端-roce)。

业务构造示例

```python
runtime = kvaf.NpuRuntime(
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    transport="roce",
    rdma_device="hrn5_4",
    rdma_port=1,
    gid_index=1,
    roce_cache_hbm=True,
)
runtime.link_to("90.96.15.111:29999")
```

当前实现开放图外 BatchPut 和 BatchGet。ACL Graph 捕获时会明确报错。图外默认异步 Get 由 Host 回调等待 RDMA worker，A5 上的同 stream 算子可见性和故障行为仍需实测。当前回调无法向 ACL stream 返回错误，因此异步 Get 失败会终止进程，避免下游使用部分写入的 HBM。联调时可用 `synchronize=True` 从调用处接收错误。带 Handle 的请求会在 `release_access_table` 时等相关传输完成再解锁 slot。

C++ 调用方可通过 `BatchPutExecutionOptions::hbm_owner` 或 `BatchGetExecutionOptions::hbm_owner` 提供稳定分配的共享引用。可用 `hbm_storage` 指定该引用持有的 HBM 子范围，省略时引用须保证 ACL 返回的整个分配有效。

Python 使用 `NpuRuntime(..., transport="roce", roce_cache_hbm=True)` 启用缓存，默认关闭。启用后，BatchPut 的源和 BatchGet 的目标都可以通过底层 Tensor storage 的引用复用 MR。共享同一 storage 的 Tensor 视图可复用该 storage 的 MR。KVAF 持有 storage 到驱逐或关闭完成，用户可以释放 Tensor 对象。使用该选项期间，同一 storage 的调用方及其别名都应保持底层分配地址和大小稳定，避免对已提交的 storage 执行会替换底层分配的 `resize_()` 或其他操作。对这些 buffer 进行原地读写仍遵循原有的 stream 同步约定。

临时 MR 仍在放行 stream 前注销。每个 session 的可复用 MR 默认最多 64 项、累计 1 GiB。缓存退役时先注销 MR，再释放持有的 storage。

### 上层显式注册句柄方案

现有 `roce_cache_hbm=True` 是后端根据 Tensor storage 自动查找和缓存 MR；业务侧不显式管理注册时机。为让上层能够提前注册长期使用的 HBM buffer，并让每次读写明确复用同一注册结果，增加显式 `HbmRegion` 生命周期接口。该接口只管理本地 NPU HBM，不改变远端 DDR region、Key/slot 元数据或 RoCE QP 的生命周期。

建议的 Python API：

```python
runtime.initialize(...)

# 初始化完成、RoCE PD 可用后，对长期复用的 HBM storage 显式注册。
input_region = runtime.register_hbm(input_tensor)
output_region = runtime.register_hbm(output_tensor)

runtime.batch_put(
    keys, input_tensor, input_offsets, input_lengths,
    hbm_region=input_region,
)
runtime.batch_get(
    access_table, indices, offsets, miss_count, output_indices, output_tensor,
    hbm_region=output_region,
)

runtime.unregister_hbm(input_region)
runtime.unregister_hbm(output_region)
runtime.shutdown()
```

`hbm_region` 应为不透明的 Runtime 句柄，不向 Python 暴露 `ibv_mr*`、`lkey`、PD 或底层地址描述符。旧的 `batch_put/get` 调用保持兼容：未传 `hbm_region` 时继续走现有自动注册/缓存路径；显式传入时必须使用该句柄指向的已注册 MR，不应静默重新注册一块未匹配的 HBM。

#### 句柄包含和校验的信息

Runtime 内部注册表中的每个 HBM region 至少关联以下状态：

| 状态 | 用途 |
| --- | --- |
| Runtime/session generation 与 NPU device ID | 拒绝跨 Runtime、跨 device 或 session 重建后复用旧句柄。 |
| 注册范围基址和长度 | 验证每一笔本地 RDMA 访问都包含在注册 MR 内。 |
| HBM backing allocation 范围 | 与 `aclrtMemGetAddressRange()` 确认的真实 ACL 分配范围比较，并管理 HAL segment 登记。 |
| storage 强引用及稳定身份 | 保证 Tensor storage 在 MR 使用期间仍存活，并区分地址复用后的新分配。 |
| 内部 MR、`lkey` 和权限 | 构造本地 SGE；Python 调用方不读取或修改这些字段。 |
| 状态和在途引用计数 | 关闭句柄时阻止新提交，并等待已有 RDMA 访问安全结束。 |

**注册范围选择：首版按 PyTorch `untyped_storage()` 的完整范围注册。** 同一 storage 的不同连续 Tensor view 因而可以共用句柄；每次传入的 view 仍必须落在该范围内，且符合 BatchPut/Get 的连续 Tensor 约束。找不到 storage 信息或无法证明 storage 位于当前 ACL allocation 内时，`register_hbm()` 应报错，不应退化成不受控的裸地址注册。注册范围还必须满足设备 MR 上限和 Runtime 的注册资源预算。

HAL 登记和 verbs MR 的范围需要继续分开处理：HAL segment 按 ACL 实际 allocation 的基址与长度登记；verbs MR 按 HbmRegion 的 storage 基址与长度注册到当前 PD。句柄针对的是本地 HBM，不能代替远端 DDR 的 `rkey`，也不会决定 Key 对应的远端地址。

#### 每次 BatchPut/Get 如何消费句柄

1. Python binding 将 `hbm_region` 解析成当前 Runtime 注册表中的 region，并检查它尚未注销、属于当前 NPU/session，且与传入 Tensor 的 storage 匹配。
2. Runtime/operator 为该次操作 pin 一个内部注册对象引用。调用在途期间，该 region 不可注销或驱逐。
3. operator 用该 region 内部保存的 MR 和 `lkey` 构造每笔 WR；**读写请求仍决定实际传输位置和长度**。例如 BatchPut 第 `i` 条 WR 的本地范围是 `input.data + input_offsets[i]`、`lengths[i]`；BatchGet 则检查每个输出目标区间在 output region 内。
4. 有效传输范围必须同时落在 Tensor view、HbmRegion 注册范围和 ACL allocation 范围内。校验失败时返回参数错误，不回退到重新注册。
5. 完成后释放本次 pin。同步 BatchPut 在 CQE 完成并返回前持有该引用；异步 BatchGet 的 job 必须将注册对象引用保留到 CQE 完成及所需设备可见性步骤结束。

显式句柄路径不得在每次 BatchPut/Get 内再次调用 `ibv_reg_mr()`。创建句柄时调用一次 `halMemRegUbSegment()`（或复用相同 allocation 的 HAL 登记）和 `ibv_reg_mr()`；注销该 region 时对应调用 `ibv_dereg_mr()` 和在最后一个引用时 `halMemUnRegUbSegment()`。WR 仍逐笔携带本地地址、长度和 `lkey`，因此复用 MR 不会导致整块 storage 都被传输。

#### 注销和 Runtime 关闭

- `unregister_hbm(region)` 将 region 标记为 closing，拒绝后续新操作；等待或排空现有操作 pin 后，注销 verbs MR，再减少 HAL allocation 的引用计数，并释放 Runtime 持有的 storage 强引用。只有注册表确认已关闭后，句柄才变成不可用状态。
- BatchPut 是同步路径，正常返回时本批 RDMA 已完成。BatchGet 可能仍在后台执行，注销必须等待相关 job 的 CQE 和完成发布步骤；调用方不能只依靠 Python 局部变量离开作用域来保证安全。
- `runtime.shutdown()` 先停止新请求并排空/停止所有 RDMA job，再关闭剩余 region，最后销毁 QP、CQ、PD 和 session。已注销或 session generation 不匹配的句柄再次传入时，应明确报错。
- 在 region 注册期间，上层不得对共享 storage 执行会更换底层分配的操作，例如可能重新分配的 `resize_()`。正常原地写入仍需遵循既有 stream 顺序约定。

#### 接入模块和验收

| 模块 | 需要增加的内容 |
| --- | --- |
| RoCE session/transport | 暴露内部 register/unregister 能力；将 `HbmRegistration` 管理为可 pin 的对象；增加只使用现有 MR 执行 WR 的入口。 |
| `RuntimeCore` / AccessOperator | 增加注册、注销及句柄校验接口；在 BatchPut/Get request 中传递解析后的内部 region 引用。 |
| Python binding | 暴露 `HbmRegion`、`register_hbm(tensor)`、`unregister_hbm(region)`，并为 BatchPut/Get 增加可选的 `hbm_region` 参数。 |
| 测试 | 覆盖注册一次、多次 Put/Get、不同 storage view、范围越界、跨 Runtime/device 使用、重复注销、storage 生命周期、Get 在途注销和 shutdown 清理。 |

代码入口已增加，但仍需在 A5/CANN/1825 真机上验收：显式注册阶段恰有一次 MR 注册；随后多次传输不再逐批调用 `ibv_reg_mr()`/`ibv_dereg_mr()`；注销后才出现相应 deregister；所有 PUT/GET 数据正确，且在途 GET 不会访问已注销的 HBM。

### 测量 BatchGet 的 HBM 注册开销

重新构建 NPU 客户端后，在已配置 RoCE 的环境中开启计时。

```bash
KVAF_ROCE_MR_TIMING=1 python tests/python/test_npu_batch_get.py --transport roce
```

集成测试默认启用 RoCE MR 缓存。设置 `KVAF_ROCE_CACHE_HBM=0` 可以用同一测试测每批注册的基线。正常复用时，重复使用 `output_a` 的批次应打印 `phase=cache_hit`。

客户端向 stderr 打印 `[KVAF_ROCE_MR]`。用同一个 `id` 配对 `phase=register` 和 `phase=deregister`。所有耗时单位都是微秒。

| 字段 | 注册阶段 | 注销阶段 |
| --- | --- | --- |
| `hal_us` | `halMemRegUbSegment()` | `halMemUnRegUbSegment()` |
| `verbs_us` | `ibv_reg_mr()` | `ibv_dereg_mr()` |
| `total_us` | MR 创建过程，包含 HAL 引用表的处理与锁等待 | MR 释放过程，包含 HAL 引用表的处理与锁等待 |

`hal_called=0` 表示本阶段没有调用 HAL 注册或注销接口，例如复用了已有的 HAL 登记。MR 缓存命中单独打印 `phase=cache_hit`。日志覆盖正确性用例、预热和正式采样。计时开启后打印本身也可能抬高端到端 `mean_ms`，测基线时请关闭此开关。

参考 [每批注册时序图](../host-roce-batchput-batchget-sequence.svg)、[MR 复用时序图](../host-roce-batchput-batchget-mr-reuse-sequence.svg)和 [Mooncake 参数面实现](https://gitcode.com/lsjhw/mooncake/blob/b1d41fcd1c8f17742c9dbf58f47d1c2279255841/mooncake-transfer-engine/src/transport/rdma_transport/ascend_parameter_plane.cpp)。
