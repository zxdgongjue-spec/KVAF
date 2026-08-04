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

`AclGraph` 与 `NpuRuntime.batch_get` 接收调用方持有的 `torch.npu.Stream`。KVAF 不创建、
不保存或销毁调用方的 stream；图 capture、replay 和生命周期仍由上层框架负责。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。
