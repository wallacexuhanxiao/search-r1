# GiGPO BM25 Search-R1 实验报告

日期：2026-07-07

## 1. 实验设置

本次实验是 BM25 Search-R1 Agent 的 GiGPO 对照训练，用来和主线 GRPO 训练结果做比较。

核心设置：

- 基座模型：`Qwen/Qwen2.5-3B-Instruct`
- 训练方法：LoRA + GiGPO-style search-agent training
- 检索器：本地 Wikipedia BM25
- Agent 输入的 BM25 文档数：`top_k = 3`
- 最多搜索轮数：2
- 环境最大步数：3
- 验证频率：每 50 step 一次
- 总训练步数：300

已经从 GiGPO 机器拉回本地的核心文件：

- 配置文件：`configs/search_r1_qwen3b_bm25_gigpo.yaml`
- 训练脚本：`scripts/train_verl_search_gigpo.sh`
- 训练日志：
  - `remote_artifacts/gigpo_20260707/logs/train_300_gigpo.pre_resume_step100.log`
  - `remote_artifacts/gigpo_20260707/logs/train_300_gigpo_resume_from100.log`
- 远端核心代码快照：`remote_artifacts/gigpo_20260707/repo_core/`

最终 checkpoint 状态：

- `global_step_300` 已保存
- `latest_checkpointed_iteration.txt = 300`
- 训练进度到达 `300/300`

## 2. 验证结果

| Step | NQ EM | HotpotQA EM | Success Rate | NQ 平均搜索次数 | HotpotQA 平均搜索次数 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.172 | 0.279 | 0.260 | 0.886 | 1.188 |
| 100 | 0.171 | **0.305** | **0.268** | 0.918 | 1.052 |
| 150 | **0.212** | 0.274 | 0.263 | 0.654 | 0.722 |
| 200 | 0.186 | 0.289 | 0.263 | 0.504 | 0.766 |
| 250 | 0.169 | 0.282 | 0.254 | 0.516 | 0.766 |
| 300 | 0.191 | 0.299 | 0.267 | 0.738 | 1.026 |

按指标选择：

- NQ 最优：`step150`，NQ EM = `0.212`
- HotpotQA 最优：`step100`，HotpotQA EM = `0.305`
- 总体 success rate 最优：`step100`，success rate = `0.268`
- 最终 checkpoint：`step300`，NQ EM = `0.191`，HotpotQA EM = `0.299`

## 3. 行为分析

GiGPO 学到的是一个更保守的工具调用策略。

关键现象：

- NQ 平均搜索次数从 step100 的 `0.918` 降到 step150 的 `0.654`，step200 进一步降到 `0.504`。
- HotpotQA 平均搜索次数从 step100 的 `1.052` 降到 step150 的 `0.722`。
- 到 step300，搜索使用量有所恢复：NQ `0.738`，HotpotQA `1.026`。

这解释了指标变化：

- step150 的 NQ EM 明显变高，可能是因为 NQ 中不少问题可以直接答或少搜索答，保守策略反而减少了无效搜索。
- step150 的 HotpotQA 明显变低，因为 HotpotQA 更依赖多跳证据，搜索次数下降会直接损害多跳推理。
- step300 的 HotpotQA 恢复到 `0.299`，但仍没有超过 step100。

所以 GiGPO 没有明显崩掉，它主要是学偏成了“少搜索”的策略。

## 4. 稳定性指标

训练过程没有看到明显的策略爆炸。

| Step | KL Loss | Entropy Loss | PG Clipfrac | Prompt Clip Ratio | Response Clip Ratio |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.007 | 0.597 | 0.002 | 0.000 | 0.000 |
| 150 | 0.005 | 0.577 | 0.002 | 0.000 | 0.026 |
| 200 | 0.018 | 0.796 | 0.000 | 0.000 | 0.000 |
| 250 | 0.014 | 0.609 | 0.002 | 0.000 | 0.000 |
| 300 | 0.026 | 0.733 | 0.003 | 0.000 | 0.025 |

结论：

- Prompt 没有被截断：`prompt clip ratio = 0.000`。
- Response 截断很少。
- KL 一直较小。
- PG clip fraction 接近 0。

因此 GiGPO 掉点不是典型的学习率爆炸或格式崩坏，更像是优化目标把策略推向了更少搜索。这个倾向对 NQ 有时有利，但对 HotpotQA 的多跳检索不利。

## 5. 与 GRPO 的对比

当前已知 GRPO 最好结果来自 step250。

| 方法 | Checkpoint | NQ EM | HotpotQA EM | Success Rate | NQ 平均搜索次数 | HotpotQA 平均搜索次数 |
|---|---:|---:|---:|---:|---:|---:|
| GiGPO | step150 | **0.212** | 0.274 | 0.263 | 0.654 | 0.722 |
| GiGPO | step300 | 0.191 | 0.299 | 0.267 | 0.738 | 1.026 |
| GRPO | step250 | 0.190 | **0.367** | **0.303** | 1.254 | 1.230 |

比较结论：

- GiGPO 有目前最高的 NQ 单点结果：step150，NQ EM = `0.212`。
- GRPO 在 HotpotQA 上明显更强：step250，HotpotQA EM = `0.367`。
- GRPO 的 overall success rate 更高：`0.303`。
- GRPO 的搜索次数更接近项目目标，即一个会主动调用 BM25 的多轮检索 Agent。

因此主结果应该使用 GRPO，GiGPO 更适合作为 ablation 或对照实验。

## 6. 推荐写法

推荐最终项目结果这样组织：

- 主模型：GRPO `step250`
- GiGPO 对照：
  - NQ-best：GiGPO `step150`
  - final checkpoint：GiGPO `step300`

可以写成：

> We further evaluated a GiGPO-style variant. It achieved a higher peak on Natural Questions, but learned a more conservative search policy and reduced tool usage. This hurt multi-hop HotpotQA performance, suggesting that maintaining sufficient search exploration is important for BM25-based multi-turn retrieval agents.

中文解释：

> 我们额外测试了 GiGPO-style 变体。它在 NQ 上取得了更高的单点峰值，但同时学到了更保守的搜索策略，平均工具调用次数下降。这种少搜索策略不利于 HotpotQA 这类多跳问题，说明在 BM25 多轮检索 Agent 中，保持足够的搜索探索对多跳 QA 更重要。

## 7. 下一步

建议接下来补齐四类材料：

1. 最终测试表：
   - Direct Answer
   - One-shot BM25 RAG
   - Prompted BM25 Search
   - GRPO best checkpoint
   - GiGPO step150 / step300

2. 检索命中率分析：
   - NQ hit@1 / hit@3 / hit@5
   - HotpotQA hit@1 / hit@3 / hit@5
   - 成功样本的 retrieval-hit
   - 失败样本的 retrieval-hit

3. 曲线图：
   - Validation EM vs step
   - Success rate vs step
   - Average search turns vs step

4. 轨迹案例：
   - 成功搜索并答对
   - BM25 没搜到证据
   - BM25 搜到证据但模型推理错
   - GiGPO 少搜索导致 HotpotQA 失败
