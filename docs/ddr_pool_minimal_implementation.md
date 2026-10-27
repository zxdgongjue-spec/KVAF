# 单 RuntimeCore、多 DDR Pool：最小实现与测试

## 1. 已实现的边界

本实现对应 [单 Core 多 Pool 设计](ddr_pool_single_runtime_core_design.md)，采用一个 RuntimeCore 管理多个独立 DDR regions：

```text
NpuRuntime / Mock Runtime
└─ 唯一 RuntimeCore
   ├─ PoolManager：Pool 与 layer 注册表、slot 分配、region/operator 生命周期
   │  ├─ Pool A → region A + operator A
   │  └─ Pool B → region B + operator B
   └─ 唯一 MetaManager
      ├─ 原有单 region DSA 接口，兼容旧调用
      └─ PoolMetadata：共享的分层 key 索引、事务和 pin
```

Pool 不包含子 RuntimeCore。多池模式按 `(Runtime owner, layer handle, key)` 隔离记录，namespace/layer_id 解析为唯一 LayerHandle。同一 Pool 的各层共享空闲 slots，相同 key 可在不同层独立存储；局部 key 不受 DSA 槽位下标限制。

单池旧接口继续使用原有固定 DSA 语义。多池模式显式通过 `initialize_pools()` 启用，不能混入缺少 layer 的旧 Put/Prepare/Free 请求。两种模式的元数据都归同一个逻辑 MetaManager 所有。

首版限定同一 Runtime 的 pools 使用同一个 CPU endpoint、本地 NPU 和后端配置，每 Pool 一个 region。独立 slot 大小和容量由 PoolSpec 指定。多 region 传输资源按 Pool 创建，未实现单注册 DDR 内的多个逻辑子池方案。

## 2. 主要实现文件

| 文件 | 作用 |
| --- | --- |
| [pool_types.h](../common/include/pool_types.h) | PoolSpec、PoolHandle、LayerHandle、LayerLayout、SlotLease、统计及分配器接口 |
| [pool_manager.h](../runtime/include/pool_manager.h)、[pool_manager.cc](../runtime/src/pool_manager.cc) | 独立 region/operator、共享容量、层绑定、句柄身份、分配回收与隔离 |
| [pool_metadata.h](../meta_manager/include/pool_metadata.h)、[pool_metadata.cc](../meta_manager/src/pool_metadata.cc) | 统一分层 key 索引、Prepare/Commit/Abort、稳定读 pin、原子批量 Free |
| [runtime.h](../runtime/include/runtime.h)、[runtime.cc](../runtime/src/runtime.cc) | 一个 Core 的池管理、请求路由、table 布局与生命周期 |
| [npu_runtime.cc](../runtime/src/npu_runtime.cc) | 复用已有 EID/候选探针，为多个 regions 分配内存，创建各 Pool 的 AIV operator |
| [pool_python.h](../bindings/include/pool_python.h) | Mock/NPU 共享的池管理绑定 |
| [python_module.cpp](../bindings/src/python_module.cpp)、[npu_python_runtime.cc](../bindings/src/npu_python_runtime.cc) | 旧 API 兼容、多池 layer 参数、由 table 取得 Get 布局 |

## 3. Python API

### 3.1 可在 CPU 上直接验证的例子

```python
import kvaf

runtime = kvaf.Runtime()
runtime.initialize_pools()
pool_a = runtime.create_pool("kv_1024", 128 * 1024, 8, 1024, 128)
pool_b = runtime.create_pool("kv_656", 128 * 656, 8, 656, 128)
layer_1 = runtime.bind_layer(1, pool_a, namespace_id=0)
layer_5 = runtime.bind_layer(5, pool_b, namespace_id=0)

payload = bytes(i % 251 for i in range(128 * 1024))
runtime.batch_put([71], payload, [0], [len(payload)], layer=layer_1)
table = runtime.prepare_access_table([71], layer=layer_1)
view = runtime.access_table_view(table)
assert view["token_bytes"] == 1024

output = bytearray(1024)
runtime.batch_get(table, [0], [127 * 1024], 1, [0], output, 1024)
assert output == payload[127 * 1024:]

runtime.release_access_table(table)
runtime.batch_free([71], layer=layer_1)
runtime.unbind_layer(layer_1)
runtime.destroy_pool(pool_a)
# pool_b 仍存在，Core 仍可使用。
runtime.shutdown()
```

`create_pool(name, slot_size, slot_count, token_bytes, tokens_per_block)` 要求非零容量和布局，且 `token_bytes × tokens_per_block <= slot_size`。padding 占 slot 容量但不计入有效数据长度。

多池 Put 要求每个 key 的写入长度恰好等于 `token_bytes × tokens_per_block`。尾 block 如需参与固定布局 Get，由上层补齐。Pool 容量为所有绑定层共同可用的 slot 数。

`get_layer(layer_id, namespace_id=0)` 返回已绑定的 handle；namespace_id 是整数。handle 的 owner/id 组合在 Runtime 重建、解绑重绑后不复用，等价地提供失效版本校验。

### 3.2 NPU 使用方式

沿用现有 NpuRuntime 构造函数和 `link_to`，随后调用 `initialize_pools()`、`create_pool`、`bind_layer`：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    batch_put_kernel_object_path=batch_put_kernel,
)
runtime.link_to("CPU_IP:28999")
runtime.initialize_pools()
```

构造函数的 token/block 参数保留用于旧接口和启动探针；多池实际 Get 布局取自对应 Host table。启动时先复用旧连通性验证流程，释放探针 region，再进入一个 Core 的多池模式。探针可能包含候选重试，不计作用户 Pool。

Put 接收一个连续 NPU source Tensor、CPU offsets/lengths，并在现有 `stream` 参数后增加可选关键字 `layer`。Prepare/Free 同样增加 `layer`：

```python
runtime.batch_put(keys, source, offsets, lengths, stream=stream, layer=layer)
table = runtime.prepare_access_table(keys, layer=layer)
device_table = runtime.access_table_view(table, device="npu")
runtime.batch_get(device_table, indices, offsets, miss_count, output_indices,
                  output, stream=stream, synchronize=True)
runtime.release_access_table(table)
runtime.batch_free(keys, layer=layer)
```

Get 不必重复传 layer；使用 table 中的归属和 token/block 布局，校验输出行宽。多池模式只支持受管理的 DeviceAccessTable，拒绝没有 Host handle 的裸 addresses/valid 路径。

`remote_region` 在多池模式下报错，改用 `pool_info(pool)` 查询。首版未暴露多池 profiling；相关旧 profiling 接口明确返回 NotImplemented。

## 4. 生命周期与错误语义

- Host API 调用按 Runtime 串行组织；并发 Host 提交和同 Pool 多 stream 并行不是首版能力。
- BatchGet 默认只入队。调用方必须保持 Tensor、stream、table、Pool 有效，并在关联设备访问完成后释放 table。
- graph 捕获前准备 table 并预热输出注册。最后一次 replay 完成并 reset graph 后，再释放 table、回收 key、解绑和销毁 Pool。
- DestroyPool 要求没有绑定层和存活/pending key；UnbindLayer 要求该层没有存活/pending key。违反条件返回 Busy。
- 显式 Shutdown 遇到仍存活的 AccessTable 返回 Busy。没有 table 后，可停止 operators、丢弃缓存 key、释放所有 regions；一个 operator 停止失败时保留该 region 供重试。
- 确认没有在途 DMA 的 Put 失败执行 AbortPut 并回收 slot；无法证明停止时隔离该 Pool，保留 pending slot，禁止新操作。其他 Pool 可继续使用。
- 设备 Get 返回失败时也隔离对应 Pool。完成状态不确定时，调用方的 source/output 等 buffers 仍须保持到排空或 teardown 完成。
- 成功停止 operator 后才删除元数据并释放 region。初始化失败且清理失败时，未发布的资源也保留到 Shutdown 重试，避免提前释放。
- NPU stream/graph 完成由调用方负责；Runtime 不推测 Host API 返回意味着设备执行结束。停止失败后的异常析构优先保留传输资源，调用方应检查并重试显式 Shutdown，避免资源遗留。

`pool_info` 暴露 free/reserved/committed/quarantined slots、bound_layers 和隔离状态：

```text
slot_count = free_slots + reserved_slots + committed_slots + quarantined_slots
```

本版未实现一个 Pool 多个 regions、在线扩缩容、跨 Pool 原子事务、多设备 loader 或进程崩溃后的远端 session 租约回收。

## 5. 测试覆盖与运行

| 测试 | 覆盖范围 |
| --- | --- |
| [runtime_pool_test.cc](../tests/runtime_pool_test.cc) | 真实 1024/656 布局；跨池/层/namespace；容量耗尽；原子 Free；slot 复用；配置和 handle；初始化/释放/停止失败；不确定 DMA 隔离；rehash 期间 pin；事务提交前不可见 |
| [test_runtime_pools.py](../tests/python/test_runtime_pools.py) | 真实 native binding；异构 slot；高位 key；共享池容量；padding；miss_count 无效后缀；随机重复读写回收；旧 handle 失效 |
| [test_npu_pools.py](../tests/python/test_npu_pools.py) | 真实 NPU/CPU 服务的两 region 写入、token 读取、同一 graph 中两种布局的 replay、单 Pool 销毁；默认显式跳过 |
| 原有默认 CTest 项 | DSA key helper、AIV kernel model/schedule、inflight 配置、EID 解析、旧 Mock API 回归 |

标准 CPU 构建：

```bash
python3 -m pip install pybind11
cmake -S . -B build-pools -DKVAF_BUILD_PYTHON=ON
cmake --build build-pools --parallel
ctest --test-dir build-pools --output-on-failure
```

Windows 的 CPU/Mock 构建不编译 POSIX PeerLink/RuntimeService；硬件角色仍要求 Linux/CANN。可使用提供 C++17 编译器、归档器和 Python development headers 的工具链。本次本机使用工作区内的 Zig/Clang、CMake、Ninja、pybind11 完成编译与回归；这些工具位于忽略目录 `.build-tools/`。

在已完成 CANN 构建且 CPU 服务可达的设备环境运行：

```bash
KVAF_RUN_NPU_TESTS=1 \
KVAF_CPU_SERVICE_ADDR=CPU_IP:28999 \
KVAF_BATCH_GET_KERNEL=/path/to/batch_get_kernels.o \
KVAF_BATCH_PUT_KERNEL=/path/to/batch_put_kernels.o \
PYTHONPATH=build-npu/python \
python3 tests/python/test_npu_pools.py
```

没有显式硬件开关时，该测试返回 77，CTest 标记为 Skipped。CPU Mock 通过不等于真实 NPU、驱动建链和 graph 已验证，报告测试结果时应分开陈述。

## 6. 本次验证结果（2026-10-27）

- 当前分支 `meta-slot`；CPU/Mock 原生库和 Python 扩展构建成功，构建目录 `build-pools-test`。
- CTest 共 9 项：8 项通过，1 项 NPU 硬件集成测试按返回码 77 明确跳过，无失败。
- 新增 C++ 测试包含 10 组场景，新增 Python 多池测试包含 10 组场景；旧 Mock、DSA、kernel model/schedule 等默认回归通过。
- 修改后的 NpuRuntime 与 NPU Python wrapper 额外使用最小 ACL 声明编译为对象文件，确认 C++ 类型和接口接线；这不是使用真实 CANN SDK 的构建验证。
- 当前机器未运行真实 CPU DDR 服务、NPU 数据传输或 ACL graph；这些由已加入的 opt-in 硬件测试覆盖，仍需在目标设备执行。
