# Full Get 与 Dense Get 共用队列的回放修复

## 结论

发现并修复一个可在 CPU 缓存模型中复现的队列一致性缺陷。投机 Dense Get 与终态 Full Get 使用同一条发送／完成队列，但两者更新持久化状态的方式不同。Full Get 的缓存写回可能覆盖已经提交到显存的发送位置。此修复同时用于原生批量 Get 和融合内联调用，没有替换 WQE 格式、传输数据、CQ 错误检查或超时门限。

这与179上“首次 Eager 成功、第二次 Eager 首批投机 Get 失败”的现象相符，但不能仅凭融合层错误码7，认定现场 KVAF 原始错误是超时或 CQ 错误。仍须核对原始状态并完成真机回放。

## 为什么第一轮完成后，第二轮才可能失败

每条队列有一条64字节的软件状态记录，包含发送位置 `sq_producer_index`、完成位置 `cq_consumer_index` 和完成批次数。它们不是互相独立的缓存行。

投机路径和公共 Post 提交函数已经使用绕过 AIV 本地数据缓存的读写：发送位置直接提交到显存。旧 Full Get 却通过普通访问读取这条状态，再在 CQ 完成后修改同一缓存行中的完成位置和计数。当这条缓存行被写回时，发送位置的旧副本也随之被写回。

CPU 回归测试调用实际 `RunKvafFullBatchGetCore`、`WaitForCompletion` 和 Dense 的 Begin/TryRetire/Finish 源函数，只将设备缓存及传输动作建模。输入让发送位置从18推进到35；旧代码在终态刷新后退回18，测试因此失败。这不是拿一个重写算法冒充真实 HCCP 仿真：测试不验证网络、带宽、WQE硬件执行，也不能给出真机时延。

## 修改

- Full Get 的初始状态、发送位置、CQ位置、完成计数和错误状态，统一使用 `ReadGmByPassDCache` / `WriteGmByPassDCache`。
- CQ消费位置先写入显存并执行屏障，再通知硬件完成队列的消费位置。
- Dense Begin 对队列初始化标志和 magic 也使用同样的访问路径。
- 不在每次回放重置队列，不伪造成功，不重新绑定传输来掩盖重入错误。

## 验证与边界

实际 Full Get 函数的 ASan/UBSan CPU回归覆盖：60轮混合调用，24轮原生调用，零行与非零行交替，CQ owner回绕，发送位置跨SQ深度，CQ错误和超时。旧代码在非零首批失败，修复后通过。

CPU模型只证明在独立缓存与显存视图下，状态更新不再倒退。CANN编译和真机连续 Eager／Graph 验证应分别记录；没有真机通过记录之前，不声明完整故障或性能目标已经解决。

本机 `cann-sim91` 已从仓库获取 `ef00d948b5ca696f736a09c513bc5a1b10da91c0`，CANN9.1 的 `kvaf_execute_batch_get_kernel` 编译通过（`/tmp/kvaf-queue-ef00d94/bin/kvaf_execute_batch_get_kernels.o`）。这只是一项编译结果，不是模拟器完成 HCCP 传输的证明。
