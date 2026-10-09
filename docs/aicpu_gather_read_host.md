# GatherRead Host 准备路径

每次 GatherRead 都使用当前请求的源地址、长度、目标偏移和长度。只复用缓冲区容量、HBM 注册和 CPU worker，不缓存上一轮的布局结果。

## 校验与准备

`runtime/src/gather_host_plan.cc` 单独按 `-O2` 编译，不改变 CPU URMA provider 或其他 transport 的编译选项。

源 span 只遍历一次，同时检查源范围、总长度、staging 和 descriptor mailbox 重叠。通过记录 staging 后最近的源起点，完成总长度累加后即可判断 staging 是否越界或覆盖源数据，不需要第二遍检查。原 direct setup 中的 mailbox/source 全量扫描也并入这一遍。

目标 span 遍历一次；升序或降序且不重叠时无需排序。任意顺序的目标复用排序缓冲区，仍逐轮检查重叠。连续输出使用栈上的一个 span，无需分配临时 vector。

`direct_setup` 只负责首次启用 CPU worker 和逐轮分配 sequence；不再扫描源清单。mailbox 校验包含在 `validate` 内。

## 合并 HBM 上传

原先源清单与目标清单分别执行一次同步 H2D。现在 URMA 模式下，输出缓冲区空间足够时，将两份清单打包后执行一次同步 H2D：

```text
HBM output
  [离散输出数据及间隙]
  [对齐到 64 B]
  [64 B 控制 scratch；分块流水模式预留 8 × 64 B 独立轮询空间]
  [48 B 源清单 header + source spans]
  [对齐到 64 B]
  [target spans]
  [动态任务计数器及 staging offsets（由 AICPU 初始化）]
```

源清单仍在已经注册的 output HBM 内，沿用原有 local token；AICPU 仅将源清单发送到 CPU DDR。目标清单和任务信息留在 HBM，不增加跨 URMA 传输的数据量。

`gather_read_storage.h` 的 `GatherReadPackedOutputBytes` 返回合并布局所需的 output 字节数，溢出返回 0。原生测试客户端已自动使用此容量，并继续使用 `ACL_MEM_MALLOC_HUGE_FIRST`。

旧调用者如果只预留了控制 scratch 和源清单空间，或 output 起点未按 64 B 对齐，会自动走原来的双上传路径；RPC 模式也保持独立上传目标清单。ACL 分配的输出满足对齐要求，没有增加必须设置的参数。

## 耗时口径

- `validate`：当前请求的源、staging、mailbox、目标布局校验。
- `direct_setup`：首次 CPU worker 启用和本轮 sequence 分配。
- `metadata_upload`：合并路径的空间判断、Host 打包和一次同步 H2D 的完整耗时。
- `source_upload`、`descriptor_upload`：旧双上传路径使用，合并路径为 0。
- HBM 注册查询、启动 AICPU、等待 stream、状态回读沿用原来的独立计时。

清单上传在 AICPU 启动前完成，没有将上传等待转移到 `stream_sync`。READ 调度、CQ 缓存维护和 CPU gather 不变。

## 运行

CPU service 已在 `127.0.0.1:28990` 运行时：

```bash
./build.sh aicpu-urma --install-kernel
./build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
  --peer 127.0.0.1:28990 --device-id 6 \
  --blocks 4096 --block-bytes 656 --output-stride 1312 \
  --sgl-size 6 --aicpu-cores 5 --sq-batch-wqes 8 \
  --read-schedule dynamic --read-task-spans 192 \
  --descriptor-transport urma --profile-detail 1 \
  --warmup 10 --iterations 1000
```

本地验证覆盖布局变更、随机有效/无效布局、越界与重叠、整数溢出、合并存储和静态/动态 READ 内核仿真。真实耗时需要在 Ascend 环境复测。
