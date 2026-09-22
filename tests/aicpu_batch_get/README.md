# AICPU 发起、CPU URMA 回推、AICPU scatter 带宽测试

这是独立 C++ 测试，不调用 Runtime BatchPut，不改 Python 或 MetaManager。

**当前 scatter 引擎是单 AICPU 核 `memcpy`，不是 SDMA。** CSV 固定标注
`aicpu_memcpy`。该版本用于验证完整链路并测量这个实现的带宽，不能据此报告
SDMA 的性能上限。Ascend/URMA 硬件验证必须在目标机器完成。

## 执行流程

1. NPU 测试进程分配/注册 HBM，向 CPU 导出 region 并处理 CreateChannel。
2. CPU 主线程初始化已有 `CpuUrmaTransport`，预注册离散源和控制缓冲区。
3. NPU 启动一次 AICPU 算子，在算子内部跑 warmup + iterations 轮。
4. 每轮 AICPU 发布 `request_seq`，CPU 用 `ReadSglBatch` 轮询它。
5. CPU 更新每行 8 字节序号，调用 `WriteBatch`，等待返回，再用
   `WriteSglBatch` 写 `done_seq`。
6. AICPU 等待序号，invalidate staging，memcpy scatter，clean 输出。
   检查各行本轮序号后才请求下一轮，不存在两轮同时使用 staging。
7. CPU 停止/销毁 transport 并确认 drained；NPU 检查输出后才释放 target。

CPU source 每行之间有 64 字节空隙，NPU staging 连续，输出按反向行序分散到
同样带 64 字节 guard 的行中。源/目标描述符固定，不包含动态 AccessTable 解析。
行长必须是 64 的倍数；一次 batch 的数据不超过配置校验允许的内存上限。

## 快速构建和运行（推荐）

在两端的仓库根目录执行。`build.sh` 自动查找 CANN/UMDK；UMDK 未安装时
沿用已有脚本逻辑获取固定版本。已有 UMDK 可以通过 `KVAF_UMDK_ROOT` 指定，
已有 liburma 可以通过 `KVAF_URMA_LIBRARY` 指定。两个新模式不构建 Python 或 AIV。

CPU 端构建：

```bash
bash build.sh aicpu-cpu
```

NPU 端加载本机 CANN 环境后构建，首次运行或重新构建 kernel 后显式安装：

```bash
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 0
```

安装需要 CANN 目录写权限；不加 `--install-kernel` 不修改安装目录。
启动脚本自动选择构建目录内的 JSON，默认端口 29991。
CPU 端先验证小规模正确性（替换机器地址、UB 设备和 EID index）：

```bash
bash tests/run_aicpu_get_cpu.sh --smoke \
  --peer NPU_HOST:29991 --ub-device UB_DEVICE --eid-index 0
```

小规模测试结束后，重新启动 NPU 脚本，再在 CPU 端测量默认 32 MiB/batch：

```bash
# NPU 端
bash tests/run_aicpu_get_npu.sh --device-id 0

# CPU 端
bash tests/run_aicpu_get_cpu.sh \
  --peer NPU_HOST:29991 --ub-device UB_DEVICE --eid-index 0 \
  --output results/aicpu_get.csv
```

两端的构建/启动脚本均支持 `--build-dir DIR`。CPU 启动脚本的 `--smoke` 设置
16 行×64 字节、预热 1 轮、测量 3 轮；显式 `--rows` 等参数可覆盖这些值。
其他 benchmark 参数原样传递。`--output` 同时显示和保存 CSV，保留测试失败退出码。
环境变量 `KVAF_AICPU_CPU_BUILD_DIR` / `KVAF_AICPU_NPU_BUILD_DIR` 可设置启动脚本
默认构建目录，默认分别为仓库下 `build-aicpu-cpu` / `build-aicpu-npu`。
自定义交叉编译器使用 `KVAF_AICPU_CXX`；安装根目录可用启动脚本的 `--cann-root` 指定。

## 手动 CMake 构建

CPU 端需要 liburma、UMDK headers 和 OpenMP：

```bash
cmake -S . -B build-aicpu-cpu \
  -DBUILD_TESTING=ON -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-aicpu-cpu --target kvaf_aicpu_get_cpu -j
```

NPU 端需要 CANN（包含 `aclrtBinaryLoadFromFile`、
`aclrtLaunchKernelWithHostArgs`）、HCCP/RA、UMDK headers，以及 CANN AICPU 交叉编译器：

```bash
cmake -S . -B build-aicpu-npu \
  -DBUILD_TESTING=ON -DKVAF_BUILD_AICPU_GET_TEST=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-aicpu-npu --target kvaf_aicpu_get_npu -j
```

编译器默认在 CANN 的 `toolkit/toolchain/hcc/bin/`；若安装布局不同，设置
`-DKVAF_AICPU_CXX=/absolute/path/aarch64-target-linux-gnu-g++`。
该开关不要求启用 AIV kernel 或 NPU Runtime service。

显式安装 device kernel（需要 CANN 目录写权限，构建不会自动安装）：

```bash
bash tests/aicpu_batch_get/install_kernel.sh \
  build-aicpu-npu/tests/aicpu_batch_get "$ASCEND_HOME_PATH"
```

脚本安装测试专属 tar/JSON 并添加 `ascend_package_load.ini` 条目。已有配置首次
修改前备份为 `.kvaf-test.bak`。设备的自定义算子加载/签名策略需符合本机 CANN
要求；脚本不改变验签配置。卸载时删除测试 tar/JSON 及该 name 对应的配置段，
不要用旧备份覆盖后来其他安装程序增加的配置。

## 运行

先启动 NPU 端：

```bash
build-aicpu-npu/tests/aicpu_batch_get/kvaf_aicpu_get_npu \
  --port 29991 --device-id 0 \
  --kernel-json build-aicpu-npu/tests/aicpu_batch_get/libkvaf_aicpu_test.json
```

默认读取 pair_info 的 NPU EID；也可以传 `--eid` 加 32 位十六进制 EID。
然后在 CPU 数据所在机器运行：

```bash
build-aicpu-cpu/tests/aicpu_batch_get/kvaf_aicpu_get_cpu \
  --peer NPU_HOST:29991 --ub-device UB_DEVICE --eid-index 0 \
  --rows 8192 --row-bytes 4096 --warmup 5 --iterations 100 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --wqes-per-doorbell 0 --timeout-ms 30000
```

先用 `--rows 16 --row-bytes 64 --warmup 1 --iterations 3` 验证，再测大 batch。
EID 需要选择当前硬件上可达的 CPU/NPU 配对；测试不自动遍历候选。
每个 NPU 测试进程仅接受一次配置/测量，下一组参数重新启动。

## 计时与结果

成功且最终数据验证通过才输出一行 CSV：

- `e2e_GBps`：有效 payload 字节数 / 平均端到端 ns，十进制 GB/s。
- `mean_ns/p50_ns/p99_ns`：AICPU 发布请求前到 scatter 完成并 clean 输出。
- `wait_ns`：请求发布、CPU 控制轮询、行序号更新、WRITE 和 done 通知。
- `scatter_ns`：staging invalidate、AICPU memcpy 和输出 cache clean。
- `cpu_write_ns/cpu_prepare_ns/cpu_post_ns/cpu_poll_ns`：现有 WriteBatch 统计。
- `mean_wrs`：仅数据 WriteBatch 的平均 WR 数；`control_reads` 包含预热的控制轮询次数。

初始化、注册、launch、warmup、每轮 tag 校验和最终逐字节 readback 不在报告的
端到端样本内。每轮 O(rows) 的 CPU tag 更新在端到端内，不在 cpu_write_ns 内。
因此该结果是完整串行握手模式的有效带宽，不是只测 post 的带宽，也不是
多批次重叠时的稳态吞吐。纯 CPU gather WRITE 基线继续使用
`kvaf_cpu_urma_eid_perf`。

## 正确性和平台限制

- 控制记录分离 256 字节，设备按 64 字节 cache line 显式 clean/invalidate；
  cntvct/cntfrq 提供设备侧计时。需要目标 AICPU 支持这些指令及其内存映射语义。
- 每轮更新/验证所有行的序号，最后验证完整 payload 和所有 guard。
- 依赖 provider 的成功 WRITE completion 保证数据对远端可见；全部数据 WR
  完成后才提交 done WRITE。测试结果通过并不能替代对 provider 内存顺序契约
  的确认。如果硬件要求额外 fence，应在得到已验证接口后加入，不能凭空假设。
- 失败不报告有效带宽。CPU 出错时停止 transport，NPU 算子超时退出。
  传输/stream 失败或 NPU 无法收到 CPU drained 确认时保留进程和 target，打印诊断；确认 CPU 已退出后
  再结束 NPU 测试进程，防止在途 WRITE 访问已释放内存。
- 异常终止/断连不是自动恢复场景；修复环境后重启两端。未实现 SDMA、双缓冲、
  graph capture、动态地址列表或 Runtime 接入。

## 本机测试（无需 CANN/URMA）

```bash
cmake -S . -B build-host-aicpu -DBUILD_TESTING=ON
cmake --build build-host-aicpu --target kvaf_aicpu_get_protocol_test -j
ctest --test-dir build-host-aicpu -R kvaf_aicpu_get_protocol --output-on-failure
```

模拟器执行生产 kernel 共用的设备状态循环，覆盖预热/轮次、延迟完成、超时、
过期数据、错误序号、反向 scatter、guard、配置溢出和带宽统计。
它不验证 AICPU 指令权限、CANN 加载或实际 UB/HBM 缓存一致性。
