# small-post 实验已结束

主动把 READ 拆成更小的 post 分给多个 Jetty 的策略已移除。历史数据中有收益、
持平和退化配置，尚无跨场景、多轮且关闭诊断后的稳定收益证据。
这不等于证明多 Jetty 永远无益；当前采用较简单的 baseline 分配。

当前规则、默认开关、完整 A/B 命令和 CSV 字段见
[baseline post 分配与按需 Jetty 管理](batch_get_lean_schedule.md)。
不再使用 `post_limit` 或 `--batch-get-small-post-max-requests`。

## 历史日志汇总

旧日志继续使用原工具，文件名保留 baseline/small-post 标签：

```bash
python3 tests/diagnostics/summarize_small_post.py logs/small-post
```

它生成六张 `small_post_*.csv`，保留历史 small-post/fused 计数和实际分配证据。
缺少的旧测量留空。请将历史日志与当前 baseline/lean 日志放在不同目录。

当前 baseline/lean 日志使用新工具：

```bash
python3 tests/diagnostics/summarize_batch_get.py logs/batch-get-lean-serial
```

内核 V3 ABI 的旧 small-post/fused 字段保留位置，当前固定写 0，便于旧格式
解析；这些兼容字段不代表策略仍可开启。
