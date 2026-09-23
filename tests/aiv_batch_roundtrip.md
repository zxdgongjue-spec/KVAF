# 原生 AIV BatchPut/BatchGet 测试

这个客户端只使用 ACL、KVAF PeerLink 和现有 Direct-HCCP AIV transport；不使用
torch_npu、Python binding，也不调用 `NpuRuntime::Initialize` 的 WRITE 预探针。
它自动按 KVAF 的顺序尝试 CPU 候选和本地 NPU EID，找到首个完整通过的组合后退出。

同机双终端：

```bash
# 终端 1：CPU
./build.sh cpu
build-cpu/runtime/kvaf_cpu_service
```

```bash
# 终端 2：NPU
./build.sh aiv-native
build-aiv-native/tests/kvaf_aiv_batch_roundtrip --device-id 0
```

CPU 服务默认监听 `127.0.0.1:28999`。异机时在 CPU 端使用
`--listen 0.0.0.0`，在 NPU 端额外传 `--peer CPU_IP:28999`。
可选参数：`--blocks`、`--block-bytes`、`--token-bytes`、`--iterations`，
以及 `--put-kernel` / `--get-kernel` 覆盖构建产物路径；`--help` 查看完整用法。
日常测试不需要指定 CPU 候选或 NPU EID。

每轮向 CPU DDR 执行 AIV BatchPut，再从同一区域执行 AIV BatchGet，反向排列
输出 token 并逐字节比对。每个候选对单独创建、销毁 transport。Put 失败时仍尝试
Get 的 CQ 完成检查，但不会比对未确认写入的 DDR 内容，也不会输出通过标记。
只有两步完成且所有字节匹配才输出 `AIV_BATCH_ROUNDTRIP_PASS`，否则退出码非零。
这个测试绕过的是 Runtime 的预探针，不是 KVAF 的 PeerLink 建链实现。
