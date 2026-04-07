# 如何生成 rule

更新时间：2026-04-02

这份说明不是泛泛讨论“KG + LLM 怎么结合”，而是专门回答你这个仓库里的 `rule` 应该怎么生成、参考哪些文献、以及最适合当前代码的落地路线。

## 1. 先明确：这个仓库里的 `rule` 是什么

先看代码语义，而不是先看名字。

- `workflow/predict_paths_and_answers.py` 和 `workflow/predict_final_answer.py` 里的 `add_rule` 都只是读取外部 `rule_path`，再把其中的 `prediction` 合并到样本的 `predicted_paths`。
- `src/qa_prompt_builder.py` 里的 `apply_rules(...)` 会把这些 `predicted_paths` 逐条送进 `utils.bfs_with_rule(...)`。
- `src/utils/graph_utils.py` 里的 `bfs_with_rule(graph, start_node, target_rule)` 会逐跳检查当前边的 `relation` 是否等于 `target_rule[len(current_path)]`。

所以，这个仓库里的 `rule` 本质上更像：

- 关系序列
- 关系模板
- relation-only path skeleton

而不是：

- 完整实体路径
- 自然语言解释
- 通用逻辑公式字符串

最符合当前代码的数据形态是：

```json
["并发症", "就诊科室"]
```

或者 Freebase 风格：

```json
["people.person.parents", "people.person.children"]
```

也就是说，你真正要生成的目标应当是：

`question + q_entity -> top-k 关系序列`

然后再让代码把这些关系序列展开成图上的真实路径。

## 2. 文献里“生成 rule”的几条主路线

截至 2026 年 4 月，和你这个需求最相关的研究大致可以分成四类。

### 2.1 直接生成关系路径计划

这类方法和你当前仓库最接近，因为它们的中间表示本来就是“关系路径”或“路径计划”。

- **RoG**（Reasoning on Graphs）提出了 planning-retrieval-reasoning 三阶段框架：先让模型生成 relation paths 作为 plan，再据此到 KG 中检索真实 reasoning paths，最后用这些路径推理答案。它和你当前 `rule -> bfs_with_rule -> reasoning paths` 的思路高度一致，是最值得直接借鉴的一类。  
  链接：<https://arxiv.org/abs/2310.01061>

- **KARPA** 是更像工程化替代方案的路线：不做额外训练，而是先由 LLM 做全局 relation path 预规划，再用 embedding 模型去匹配图中语义相关路径，最后聚合这些路径作答。它适合“先快速做出来一个可用版本，再决定要不要训练”的场景。  
  链接：<https://aclanthology.org/2025.findings-acl.1269/>

- **ToG** 更偏“在线搜索”而不是“离线产出规则库”。它让 LLM 在图上迭代式地探索实体和关系，做 beam search，逐步保留最有前景的轨迹。ToG 不一定直接给你一个固定 `rule_path` 文件，但它非常适合生成高质量搜索轨迹，然后再把轨迹压缩成关系序列，作为后续 rule 数据来源。  
  链接：<https://arxiv.org/abs/2307.07697>

- **RJE** 提出 retrieve-judge-explore：先取回一批 reasoning paths，再判断这些证据是否已经足够，不够再继续探索。它的重要启发是：不要只看“某条 rule 能不能执行”，还要看“这些 rule 展开的证据够不够回答问题”。这对多答案问题尤其重要。  
  链接：<https://aclanthology.org/2025.emnlp-main.873/>

这一类方法的共同点是：`rule` 不是最终答案，而是“搜索计划”或“检索控制信号”。

### 2.2 先生成逻辑形式，再映射成 rule

这类方法不一定直接预测关系序列，但会先生成更高层的中间表示，再把它还原到 KB/KG 的关系与实体上。

- **ChatKBQA** 走的是 generate-then-retrieve：先用微调后的 LLM 生成 logical form，再用无监督检索替换逻辑式中的实体和关系。对你来说，一个很自然的变体是：不一定输出完整 logical form，而是只抽其中的 relation skeleton，当成 rule。  
  链接：<https://aclanthology.org/2024.findings-acl.122/>

- **RGR-KBQA** 采用 Retrieve-Generate-Retrieve：先用 KG 帮助语义理解，再生成逻辑形式，最后做实体/关系检索修正。它的启发是，关系序列可以不必一开始就完全由模型“盲写”，可以先注入局部 KG 候选关系，再生成。  
  链接：<https://aclanthology.org/2025.coling-main.205/>

- **ORT**（Ontology-Guided Reverse Thinking）不是直接从起点实体向外走，而是先抽取问题的 purpose label 和 condition label，再依据 ontology 从“目标端”反向构造 label reasoning paths，最后再用这些路径指导知识检索。它适合复杂问题，尤其适合你担心“路径虽然合法，但和问题不相关”的情况，因为它先约束目标语义，再去找路径。  
  链接：<https://aclanthology.org/2025.acl-long.741/>

这一类方法更适合复杂 schema、复杂组合查询，优点是中间表示更抽象，缺点是落到你当前仓库时，往往还要多做一步“映射成 relation sequence”。

### 2.3 先分解问题，再为每个子问题生成短 rule

如果问题很复杂，直接让模型产一条长 relation sequence 往往不稳定。这时更合理的做法是先拆问题。

- **KELDaR** 使用 question decomposition tree，把复杂问题拆成 atomic questions，再分别做 atomic retrieval。对你的启发是：不是每个问题都该生成一条长 rule；很多时候更适合先拆成多个子问题，每个子问题只生成 1-hop 或 2-hop 的短 rule，最后再组合答案。  
  链接：<https://aclanthology.org/2024.findings-emnlp.670/>

- **RJE** 的 question decomposition 模块也支持这种思路：先判断当前证据是否不足，再分解并继续探索。

这类方法特别适合：

- 多跳问题
- 多约束问题
- 多答案问题

因为它不会把所有希望都压在一条路径上。

### 2.4 显式构建规则库

如果你想做的是“可解释、可复用”的 rule 系统，而不是每次都临时让 LLM 即兴规划，那么规则库思路更合适。

- **Rule-KBQA** 非常值得参考，因为它真的在讨论“规则怎么来”。它把流程分成 induction 和 deduction 两阶段：先从已有数据中抽规则，再让 Rule-Following Fine-Tuned LLM 生成额外规则，形成 rule library；推理时再由符号 agent 在规则指导下逐步构造可执行 logical form。  
  链接：<https://aclanthology.org/2025.coling-main.562/>

对你而言，这篇工作的价值不在于直接复现 logical form agent，而在于它说明了一件事：

- rule 可以先从训练数据中抽出来
- 再用 LLM 对规则库做扩展
- 最后把 rule 当成稳定中间层，而不是每次都从头生成

如果你的数据集领域固定、关系集合相对稳定，这条路线很有吸引力。

## 3. 哪条路线最适合你现在这个仓库

如果目标是“尽量少改代码，尽快生成能被 `add_rule` 直接消费的文件”，我建议优先级如下。

### 方案 A：LLM 直接生成 top-k 关系序列，再用图执行过滤

这是最适合当前仓库的起步方案。

流程如下：

1. 从 `q_entity` 出发，枚举局部子图中的候选关系集合。
2. 给 LLM 输入问题、主题实体、候选关系、最大 hop 数。
3. 让它输出 `top-k` 个关系序列。
4. 对每个关系序列调用 `bfs_with_rule(...)` 执行。
5. 删除执行后为空的 rule。
6. 保留可执行、且彼此有差异的 rule，写入 `rule_path`。

这条路线最接近：

- RoG 的 “先 plan 再 retrieve”
- KARPA 的 “先预规划 relation paths 再匹配”

优点：

- 不需要先改主推理代码。
- 生成结果能直接被 `add_rule` 使用。
- 可以很容易扩展到多答案场景，只要保留多条 rule 即可。

缺点：

- 如果不给候选关系集合，LLM 容易幻觉关系名。
- 如果只保留一条最高分 rule，仍然可能漏掉其他正确答案。

### 方案 B：从训练集真值路径中抽 relation sequence，先做 SFT，再做 RL/GRPO

如果你已经在项目里做过 SFT 和 GRPO，而且效果不错，那么这条路线很自然。

具体做法：

1. 从训练集的 gold path 或可执行路径中抽取 relation-only 序列。
2. 对一个问题保留多条正例 rule，而不是只保留一条。
3. 训练一个小模型或中模型，输入 `question + q_entity`，输出 `top-k rule`。
4. 在 SFT 后再做偏好优化或 RL/GRPO，奖励函数不只看答案是否命中，也看：
   - rule 是否可执行
   - 展开后是否能覆盖更多正确答案
   - rule 是否和问题相关
   - 多条 rule 之间是否足够多样

这条路线最接近：

- RoG 的 relation plan 学习
- Rule-KBQA 的规则诱导

如果你后面真的想把 `rule` 做成稳定模块，这可能是长期最强方案。

### 方案 C：先做问题分解，再为每个子问题生成短 rule

如果你的问题经常是：

- 多跳
- 多条件
- 多答案

那么不要让模型直接产一条长 rule，更合理的是：

1. 先把问题拆成若干 atomic sub-questions。
2. 每个子问题只生成很短的 rule，例如 1-hop 或 2-hop。
3. 在图上分别执行。
4. 再做交集、并集或答案聚合。

这条路线最接近：

- KELDaR 的 decomposition tree
- RJE 的 sufficiency-aware exploration
- ORT 的“先确定目标语义，再反向构造路径”

如果你未来重点想解决“一个问题对应多个正确答案、多条正确路径”的情况，我反而更推荐你把这个方案作为中期方向。

## 4. 我最推荐的实际落地版本

如果只选一条，我建议你先做下面这个版本：

### 4.1 输入

- 问题文本
- 主题实体 `q_entity`
- 从主题实体出发、在 `hop <= L` 子图内收集到的候选关系集合
- 最大路径长度 `L`
- 需要输出的 rule 数量 `k`

### 4.2 LLM 生成目标

让模型输出：

- `k` 条关系序列
- 每条长度不超过 `L`
- 只允许使用候选关系集合中的关系
- 尽量覆盖不同可能答案，而不是只保留一条最短路径

### 4.3 图执行与筛选

把生成的每条 rule 用 `bfs_with_rule(...)` 展开成真实路径，然后根据下面规则过滤：

- 删除不可执行 rule
- 删除展开后完全重复的 rule
- 对多答案任务，保留能导向不同尾实体集合的 rule
- 若只剩空集，则回退到 DFS 或回退到不带 rule 的搜索

### 4.4 为什么这个版本适合当前仓库

因为它和现有 `add_rule` 的数据流完全一致：

- 外部文件提供 relation sequences
- `apply_rules(...)` 把 relation sequences 展开成真实路径
- 路径生成阶段可把展开结果写进 trie
- 最终答题阶段可把展开路径放进 `Reasoning Paths:` 上下文

也就是说，这个方案不是“重做系统”，而是给现有系统补上“rule 从哪来”。

## 5. 一个可直接使用的生成模板

下面这个模板是为当前仓库量身定做的，不是论文原文，而是综合 RoG、KARPA、RJE 这类方法后给出的工程化版本。

```text
你是知识图谱关系路径规划器。

给定：
1. 一个问题
2. 主题实体
3. 候选关系集合
4. 最大跳数 L

请输出最可能帮助回答问题的 top-k 条关系序列。

要求：
1. 每条序列长度为 1 到 L
2. 只能使用候选关系集合中的关系
3. 不要输出实体名
4. 不要解释
5. 优先保留能够覆盖不同潜在答案的多样化序列
6. 输出 JSON 数组，例如：
   [["并发症"], ["并发症", "就诊科室"], ["常见症状"]]
```

为了减少幻觉，候选关系集合最好来自局部图，而不是让模型自由生成。

## 6. 训练数据怎么构造

如果你想做 SFT 或后续 RL，可以按下面方式构数据。

### 6.1 正例 rule

从真值路径或已知可执行路径中抽 relation-only 序列。

例如真实路径是：

```text
糖尿病 -> 并发症 -> 糖尿病肾病 -> 就诊科室 -> 肾内科
```

则对应正例 rule 可以写成：

```json
["并发症", "就诊科室"]
```

一个问题如果有多条真值路径，就保留多条正例 rule，不要强行压成一条。

### 6.2 难负例 rule

难负例比随机负例更重要。可以优先构造：

- 可执行，但导向错误答案的 rule
- 和正确 rule 只差一步关系的 rule
- 语义上看起来相关，但无法支持答案的 rule

这样训练出来的模型会更会区分“可执行”和“真相关”。

### 6.3 如果做 RL/GRPO，奖励应看什么

建议至少包含四项：

- **Executability**：rule 能否在图上展开出非空路径
- **Answer coverage**：展开路径能否覆盖正确答案，尤其是多个正确答案
- **Question relevance**：展开路径是否真正和问题意图相关
- **Diversity**：多条 rule 是否覆盖不同路径簇、不同尾实体簇

这比只看“最终答案是否对”更适合多答案 KGQA。

## 7. `rule_path` 文件建议格式

当前代码里，`merge_rule_result(...)` 会读取：

- `id`
- `prediction`
- `ground_paths`

所以建议每一行至少长这样：

```json
{"id":"demo_1","prediction":[["并发症","就诊科室"],["常见症状"]],"ground_paths":[]}
```

字段说明：

- `id`：和原始 QA 样本对齐
- `prediction`：`List[List[str]]`，每个元素是一条 relation sequence
- `ground_paths`：如果你没有现成 gold path，可以先写空列表；但如果后续脚本开启 `filter_empty`，空列表样本可能会被过滤掉

如果你想保存额外分数，也可以扩展为：

```json
{
  "id": "demo_1",
  "prediction": [["并发症", "就诊科室"], ["常见症状"]],
  "ground_paths": [],
  "scores": [0.92, 0.77],
  "meta": {
    "planner": "qwen_relation_planner",
    "max_hop": 2
  }
}
```

主流程目前只会直接消费 `prediction` 和 `ground_paths`。

## 8. 对多答案问题，要怎样生成 rule

这是你场景里非常关键的一点。

如果一个问题可能有多个正确答案，rule 生成策略不应该是：

- 找到一条能执行的路径就停止

而应该是：

- 生成多条可执行且彼此不同的 rule
- 展开所有 rule，对应得到多簇 reasoning paths
- 只在“新增 rule 已经不再带来新增候选答案或新增证据”时再停止

这里最值得借鉴的是：

- ToG 的多分支 beam search
- RJE 的 sufficiency judgment
- KELDaR 的分解式处理

所以，如果你的目标是“回答多个答案”，比起单条最优 rule，更应该追求：

- top-k 多样 rule
- 多条真实 reasoning paths
- 覆盖多个尾实体簇

## 9. 当前代码层面的注意事项

### 9.1 `add_rule` 目前只消费 rule，不生产 rule

当前仓库并没有完整的“rule 生成 workflow”。现有代码只负责：

- 读取外部 `rule_path`
- 合并到数据样本
- 在图上按规则展开

所以你需要单独准备一个“rule generator”。

### 9.2 路径生成阶段和最终答题阶段对 `rule` 的使用方式不同

- 在 `workflow/predict_paths_and_answers.py` 中，`add_rule=True` 时会优先用 rule 在图上展开真实路径，再把这些路径放进 trie；如果 rule 为空，会回退到 DFS。
- 在 `workflow/predict_final_answer.py` 中，`add_rule=True` 时会把 rule 先展开成 reasoning paths，再作为 `Reasoning Paths:` 上下文拼进 prompt。

所以 `rule` 不是直接答案，而是控制“证据怎么被找出来”的中间层。

### 9.3 当前代码里有一个需要特别注意的不一致

`src/utils/graph_utils.py` 的 `bfs_with_rule(...)` 期望 `rule` 是关系序列；
但 `workflow/predict_final_answer.py` 在 `\"gcr\" in args.model_name` 的分支里，又把 `predicted_paths` 当成路径字符串，并对每条样本执行：

```python
p.split(" -> ")
```

这意味着：

- 如果你的 `rule_path` 里存的是 `List[List[str]]` 形式的关系序列
- 同时你在最终答题阶段又使用了 `gcr` 命名的答案模型分支

那么这部分代码在语义上是不一致的，可能需要额外修补。

更稳妥的做法是：

- 要么先用非 `gcr` 的最终答题分支
- 要么把这段逻辑改成“先把 rule 展开成 reasoning paths，再从真实路径里收集实体”

### 9.4 如果你后面要做真正的 rule 系统，GCR 本身不是 rule 生成方法

GCR 的核心是：

- KG-Trie
- graph-constrained decoding
- 小模型图上路径生成 + 大模型归纳式答题

它解决的是“如何让生成过程严格受图约束”，而不是“如何生成 rule”。所以你现在问“rule 从哪里来”，本质上是在给 GCR 风格系统补一个上游 planner/inducer 模块。

## 10. 一个简洁结论

如果你现在马上要做一个能跑起来的版本，我建议：

1. 先从主题实体的局部图中抽候选关系集合。
2. 用 LLM 直接生成 `top-k` 关系序列。
3. 用 `bfs_with_rule(...)` 执行并过滤空 rule。
4. 保留多条可执行且多样的 rule，不要只保留一条。
5. 把结果写成 `jsonl` 给 `add_rule` 使用。

如果你后面想把这件事做强，再升级到：

1. 从训练数据抽 relation sequences 做 SFT。
2. 用 RL/GRPO 优化可执行性、答案覆盖率和多样性。
3. 逐步沉淀成规则库，朝 Rule-KBQA 那种“规则诱导 + 规则执行”方向发展。

## 参考文献

1. Sun, J. et al. **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph**. ICLR 2024. <https://arxiv.org/abs/2307.07697>
2. Luo, L. et al. **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**. ICLR 2024. <https://arxiv.org/abs/2310.01061>
3. Luo, L. et al. **Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models**. ICML 2025. <https://proceedings.mlr.press/v267/luo25t.html>
4. Xu, Z. et al. **ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models**. Findings of ACL 2024. <https://aclanthology.org/2024.findings-acl.122/>
5. Wang, Y. et al. **A Framework of Knowledge Graph-Enhanced Large Language Model Based on Question Decomposition and Atomic Retrieval**. Findings of EMNLP 2024. <https://aclanthology.org/2024.findings-emnlp.670/>
6. Li, Y. et al. **KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model’s Reasoning Path Aggregation**. Findings of ACL 2025. <https://aclanthology.org/2025.findings-acl.1269/>
7. Zhang, Y. et al. **Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering**. ACL 2025. <https://aclanthology.org/2025.acl-long.741/>
8. Li, X. et al. **RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs**. EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.873/>
9. Zhang, Z. et al. **Rule-KBQA: Rule-Guided Reasoning for Complex Knowledge Base Question Answering with Large Language Models**. COLING 2025. <https://aclanthology.org/2025.coling-main.562/>
10. Feng, T. and He, L. **RGR-KBQA: Generating Logical Forms for Question Answering Using Knowledge-Graph-Enhanced Large Language Model**. COLING 2025. <https://aclanthology.org/2025.coling-main.205/>
