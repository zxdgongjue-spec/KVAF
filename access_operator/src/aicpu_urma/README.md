# AICPU URMA BatchPut / BatchGet

这个可选后端不创建第二条链路：沿用 KVAF `NpuRuntime::LinkTo` 的候选 EID/PeerLink
选择，以及现有 Direct-HCCP Host 侧的 HBM/Host DDR 注册、远端导入、USER_CTL
Jetty、SQ/CQ 和 token。AICPU 只从现有 transport 的 entity/queue 表生成并提交
WRITE（BatchPut）或 READ（BatchGet）WQE，等待 CQ，写回结果。

目前采用**单 AICPU worker、单 Jetty**，每 32 个请求提交一次带 completion 的
批次；AIV 模式保持默认不变。BatchPut 同步等待并验证状态；BatchGet 保留调用方
`synchronize` 语义，开启同步时检查 AICPU 状态。不支持与同一 Jetty 上 AIV
kernel 并发使用；异步 BatchGet 需要调用方在释放输出/transport 前同步 stream。

## 构建与部署

在有 CANN toolchain 的 NPU 主机上：

```bash
./build.sh aicpu-urma
bash access_operator/src/aicpu_urma/install_kernel.sh \
  build-aicpu-urma/access_operator/src/aicpu_urma "$ASCEND_HOME_PATH"
```

安装写入 CANN 目录和 package-load 配置，需要相应权限；不在 CMake 构建期间自动安装。
原生客户端内部传入 AICPU JSON 并使用单 worker、单 Jetty；不需要 AIV kernel
object 路径。服务端仍是 KVAF 原有 CPU Runtime 服务。

Host 单测：`ctest --test-dir build-aicpu-urma -R kvaf_aicpu_urma_ --output-on-failure`。
无需 CANN 的机器也可运行：`cmake -S . -B build-host -DBUILD_TESTING=ON && cmake --build build-host --target kvaf_aicpu_urma_wqe_test kvaf_aicpu_urma_kernel_test kvaf_aicpu_urma_client_options_test && ctest --test-dir build-host -R kvaf_aicpu_urma_ --output-on-failure`。

真实 CPU/NPU 回读测试：CPU 端先运行 `./build.sh cpu` 和
`build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999`；
NPU 端安装内核并重启测试进程后执行：

```bash
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 \
  --kernel-json "$ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json" \
  --device-id 0 --blocks 16 --block-bytes 4096 --token-bytes 256 \
  --warmup 1 --iterations 5 --seed 123
```

`--blocks` 为每批 KV block 数；`--block-bytes` 为每个 block 的写入字节数，
必须能被 `--token-bytes` 整除。客户端会对每个 block 的所有 token 执行 BatchGet，
反向排列输出并逐字节比对，因此覆盖完整 BatchPut payload 和 BatchGet READ。
`--ddr-bytes` 可选，至少为 `blocks * block-bytes` 且是 block 的整数倍。
每轮先写入新 payload，再读取并验证；warmup 不计入统计，输出每轮耗时和平均
`put_GBps/get_GBps`。Put 计时包含 MetaManager 的预留和提交；Get 计时包含
Runtime 提交与 stream 同步，不包含 AccessTable 准备、HBM 上传/回读和逐字节校验，
因此不是纯 SQ 带宽。
失败时退出码非零且不会输出 `AICPU_URMA_PASS`；`--help` 查看完整参数。

**硬件验证状态**：本分支的 Host WQE ABI/模拟队列测试和内核语法检查不等价于 NPU
收发验证。部署后应先用 Runtime 自带的 LinkTo WRITE 探针确认链路，再在真实
CPU DDR/HBM 之间做小 batch 的 put/get 数据回读验证，最后再进行性能测试。
