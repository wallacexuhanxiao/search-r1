# Search-R1 BM25 Final Test Report

日期：2026-07-07

## 1. 测试设置

本报告汇总最终 held-out test split 上的结果。test split 共 1000 条：

- Natural Questions: 500
- HotpotQA: 500

评测方法：

- Direct Answer: `Qwen/Qwen2.5-3B-Instruct`，不允许搜索。
- GRPO step250: LoRA-GRPO BM25 Search Agent，使用 validation 上选择出的最佳 checkpoint。
- GiGPO step150: GiGPO 对照模型的 NQ validation 峰值 checkpoint。
- GiGPO step300: GiGPO 完整训练 300 step 后的 final checkpoint。

Agent 设置保持一致：

- BM25 top-k 输入：3
- 最多搜索轮数：2
- 环境最大步数：3
- 最大 prompt 长度：2048
- 最大 response 长度：384
- 检索接口：本地 Wikipedia BM25 `/retrieve`

本地结果文件：

- Direct Answer: `remote_artifacts/test_20260707/grpo_machine/logs/direct_answer_test.summary.json`
- GRPO step250: `remote_artifacts/test_20260707/grpo_machine/logs/test_grpo_step250.log`
- GiGPO step150: `remote_artifacts/test_20260707/gigpo_machine/logs/test_gigpo_step150.log`
- GiGPO step300: `remote_artifacts/test_20260707/gigpo_machine/logs/test_gigpo_step300.log`
- BM25 test hit rate: `remote_artifacts/test_20260707/gigpo_machine/logs/retrieval_recall_audit_test_summary.json`

## 2. 最终 Test 结果

| 方法 | Checkpoint | NQ EM | HotpotQA EM | Overall / Success | NQ Success | HotpotQA Success | NQ 平均搜索次数 | HotpotQA 平均搜索次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct Answer | base | 0.112 | 0.164 | 0.138 | - | - | 0.000 | 0.000 |
| GRPO | step250 | 0.203 | **0.339** | **0.295** | 0.224 | **0.361** | **1.168** | **1.180** |
| GiGPO | step150 | 0.180 | 0.289 | 0.248 | 0.209 | 0.283 | 0.678 | 0.684 |
| GiGPO | step300 | **0.212** | 0.315 | 0.283 | **0.228** | 0.328 | 0.772 | 1.014 |

Direct Answer 的 answer rate：

| Dataset | Answer Rate |
|---|---:|
| All | 0.987 |
| NQ | 0.974 |
| HotpotQA | 1.000 |

说明：

- `NQ EM` 和 `HotpotQA EM` 是按数据集分别统计的 exact match。
- `Overall / Success` 是整体 test set 上的 success rate。
- Direct Answer 没有工具调用，所以搜索次数固定为 0。
- Direct Answer 的日志没有 `nq_success_rate` / `hotpotqa_success_rate` 字段，因此表中用 `-` 标记。

## 3. BM25 Test Hit Rate

对 test split 使用原问题直接检索 BM25，统计 gold answer 是否出现在 top-k 文档中。

| Dataset | Total | Hit@1 | Hit@3 | Hit@5 | Empty Rate |
|---|---:|---:|---:|---:|---:|
| All | 1000 | 0.316 | 0.446 | 0.503 | 0.000 |
| NQ | 500 | 0.260 | 0.410 | 0.482 | 0.000 |
| HotpotQA | 500 | 0.372 | 0.482 | 0.524 | 0.000 |

检索延迟：

- 平均延迟：43.06 ms
- P95 延迟：107.93 ms

## 4. 主要结论

GRPO step250 是当前最佳主模型。

- 相比 Direct Answer，GRPO step250 明显提升：
  - NQ: `0.112 -> 0.203`
  - HotpotQA: `0.164 -> 0.339`
  - Overall: `0.138 -> 0.295`
- 相比 GiGPO step300，GRPO step250 的 overall success 更高：
  - GRPO step250: `0.295`
  - GiGPO step300: `0.283`
- GRPO 在 HotpotQA 上更强：
  - GRPO step250: `0.339`
  - GiGPO step300: `0.315`

GiGPO step300 是 GiGPO 的更好 test checkpoint。

- GiGPO step300 相比 GiGPO step150 全面提升：
  - NQ: `0.180 -> 0.212`
  - HotpotQA: `0.289 -> 0.315`
  - Overall: `0.248 -> 0.283`
  - NQ 搜索次数：`0.678 -> 0.772`
  - HotpotQA 搜索次数：`0.684 -> 1.014`
- GiGPO step300 在 NQ 上略高于 GRPO step250：
  - GiGPO step300: `0.212`
  - GRPO step250: `0.203`

但 GiGPO step300 仍不应作为主模型，因为 HotpotQA 和 overall 不如 GRPO step250。

## 5. Search 行为分析

GRPO 的搜索策略更积极：

- GRPO step250:
  - NQ 搜索次数：1.168
  - HotpotQA 搜索次数：1.180
- GiGPO step300:
  - NQ 搜索次数：0.772
  - HotpotQA 搜索次数：1.014
- GiGPO step150:
  - NQ 搜索次数：0.678
  - HotpotQA 搜索次数：0.684

这解释了 HotpotQA 上的差距。HotpotQA 更依赖多跳证据和检索探索，GRPO 保持更高的搜索频率，因此在 HotpotQA 上取得更好表现。

GiGPO step300 相比 step150 搜索次数上升，HotpotQA 也随之恢复，这进一步支持“搜索不足会损害多跳 QA”的判断。

## 6. BM25 瓶颈分析

BM25 test hit@3 只有：

- Overall: 0.446
- NQ: 0.410
- HotpotQA: 0.482

这意味着即使使用原问题作为 query，top-3 检索结果中包含 gold answer 的比例也不到一半。对于模型生成 query 的多轮检索场景，搜索动作的收益会更高方差：

- 搜索可能带来正确证据；
- 搜索也可能带来噪声 passage；
- final-answer-only reward 不提供 query 级或 evidence 级中间反馈；
- 模型只能从最终答案对错中间接学习搜索行为。

因此 GiGPO 学到更保守的搜索策略是合理现象：在 BM25 top-3 召回有限时，搜索动作可能被优化过程视为高风险动作。GRPO 在本实验中保留了更多搜索探索，因此更适合 HotpotQA。

## 7. 推荐最终表述

英文：

> On the held-out test set, the GRPO-trained BM25 Search Agent achieved the best overall performance, improving over Direct Answer from 0.138 to 0.295 success rate. GiGPO step300 achieved the best NQ EM, but underperformed GRPO on HotpotQA and overall success. The BM25 test hit@3 was only 0.446, indicating that sparse retrieval quality is a key bottleneck. Under final-answer-only rewards, this limited recall makes search actions high-variance; GiGPO therefore tends to learn a more conservative tool-use policy, while GRPO maintains more search exploration and performs better on multi-hop QA.

中文：

> 在 held-out test set 上，GRPO-trained BM25 Search Agent 取得了最佳整体表现，将 Direct Answer 的 overall success 从 0.138 提升到 0.295。GiGPO step300 在 NQ 上最高，但 HotpotQA 和 overall success 仍低于 GRPO。BM25 test hit@3 只有 0.446，说明稀疏检索质量是系统瓶颈。在 final-answer-only reward 下，有限召回会让搜索动作产生高方差反馈，GiGPO 因而更容易学到保守的少搜索策略；而 GRPO 保持了更多搜索探索，因此在多跳 HotpotQA 上更强。

## 8. 当前推荐

- 主结果：GRPO step250
- GiGPO 对照：GiGPO step300
- 消融分析：GiGPO step150 vs step300，展示搜索次数恢复后 test 指标同步提升
- 瓶颈分析：报告 BM25 test hit@1/3/5 和检索延迟
