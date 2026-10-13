# GatherRead READ WQE 模板

每轮 READ 中，每个 AICPU worker 根据自己的 endpoint 和本地 HBM token，准备一次模板和私有编码缓冲区。动态任务之间复用，下一轮重新准备，不缓存请求地址。

模板预填 READ opcode、TP ID、远端 EID/token、SGE 数量、各 SGE 的 local token 和零填充；普通 WQE 与请求 completion 的 WQE 分别使用预计算的控制字。SGE 数量 1～6 各有一份布局。

编码缓冲区仍按连续 WQEBB 排列，一组最多 64 条 WQE。首次使用某个位置时复制对应模板；以后相同 SGE 布局只更新 owner/completion、远端 staging 地址、HBM 目标地址和长度。长度从当前请求取得，不假设固定 656 B。

如果某个位置的 SGE 数量发生变化，该位置及后续位置的模板有效性全部失效；后续逐条重新初始化。这同时处理 WQEBB 数量变化导致的偏移移动，以及相同 WQEBB 数量内 SGE 数量变化产生的旧 token/填充残留。新批次始终从缓冲区头部开始顺序编码。

初始化仅在首次使用或布局变化时清零/复制固定字段。SQ 写入量、分批方式、owner 翻转、completion 规则、doorbell 和 CQ 处理保持原有实现；编码缓冲区写完后仍完整拷贝到 SQ。

模板构建计入 `scatter_init`（日志 `Worker/task initialization`）和 READ wall time。模板首次套用、变量更新计入 `encode`。不会把模板准备从完整 READ 耗时中排除。

验证包括：1～6 SGE 随机切换、批次伸缩、不同 endpoint/token、变化的地址与长度、owner/completion，以及复用编码与原编码的逐字节比较。内核仿真继续检查 SQ 回绕、静态/动态任务和多 worker 数据正确性。实际性能需在 Ascend 上比较 `encode` 和整个 READ wall time；本地编码测试不能替代设备性能测试。
