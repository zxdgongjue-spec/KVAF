# Python Bindings

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

当前 Binding 用于验证以下链路：

```text
C++ 编译 → Python 原生扩展 → Python 导入
```

CPU 构建提供进程内 Mock Runtime。启用 `KVAF_ENABLE_AIV_BATCH_GET=ON` 时，额外提供
`NpuRuntime` 与 `AclGraph`：后者是 Python-owned 的 ACL Runtime Graph 薄封装，直接
暴露 capture / replay / synchronize / destroy 生命周期，用于 Python 集成测试或上层
Adaptor 的图编排；它不保存 KVAF 请求、Tensor 或 Operator 状态。

`AclGraph.copy_host_to_device_async(destination, destination_bytes, source, stream)` 是图
回放前的原生 ACL H2D 输入更新：调用方在已 capture 的同一 stream 上，把 host staging
buffer 写入地址稳定的 NPU tensor，再调用 `replay`。`source` 的所有权仍属于 Python，必须
至少保持到同一 stream 的 `synchronize` 返回。该方法不记录 Tensor、不会分配 buffer，也不
属于 Runtime / AccessOperator 生命周期。

`AclStream` 是 Python-owned 的 `aclrtCreateStream` 包装。它用于需要与 C++ AIV demo
严格同构的 raw ACL 图测试；可与 `AclGraph` 和 `NpuRuntime.batch_get` 一起使用。调用方应在
销毁图、关闭 Runtime 后调用 `close()`。`torch.npu.Stream` 仍然可以传入这些接口，用于验证
框架 stream 互操作。

`NpuRuntime(..., retain_unsynchronized_launch_args=True)` 是图诊断开关。它只对
`synchronize=False` 的 BatchGet 保留 `aclrtLaunchKernelWithHostArgs` 的参数记录至
Runtime shutdown，用来验证外部 graph capture 是否依赖 host 参数的跨 replay 生命周期。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。
