# KG + LLM 推理问答综述

更新日期：2026-04-01

本文聚焦“知识图谱（KG）与大语言模型（LLM）结合，用于知识驱动的推理问答（KGQA / reasoning QA / trustworthy reasoning）”这一方向，重点回答一个和当前仓库高度相关的问题：

> 为什么像 KG-Trie / 前缀树约束解码这样的方案，虽然能保证生成路径在图上合法，却仍然会出现“路径可达但和问题无关”的现象？文献里还有哪些更有效的替代或补充思路？

本文的核心结论是：近两年的主线已经从“只保证路径合法”逐步转向“合法 + 相关 + 可验证 + 可自纠”。因此，对于当前仓库所实现的 GCR / KG-Trie 路线，更可行的方向通常不是放弃约束解码，而是在其外侧增加问题相关性建模、分步验证和自适应探索。

## 1. 方法谱系概览

| 范式 | 代表工作 | 核心思想 | 对“可达但不相关”的帮助 |
| --- | :-- | --- | --- |
| KG 事实提示 / KG-RAG | KAPING、AMAR、GNN-RAG [2][8][12] | 先从 KG 中取相关事实、路径或子图，再交给 LLM 推理 | 中等，重点在过滤噪声和提升证据相关性 |
| 逻辑形式 / 计划优先 | StructGPT、ChatKBQA、KELDaR [1][5][7] | 先把“要找什么”结构化，再去图中执行 | 很强，能明显减少无目标扩展 |
| Agent 式图搜索 / 交互式导航 | ToG、RoG、FiDeLiS、ORT、RJE [3][4][11][13][14] | 让 LLM 逐步在图上做检索、判断、扩展与纠错 | 很强，直接解决“走错边但仍合法” |
| 图约束解码 | GCR、DoG [9][10] | 在生成时把图结构写成约束，保证链条合法 | 很强于 faithfulness，较弱于 relevance |
| 训练式全局策略学习 | GraphWalker、KG-Hopper、S-Path-RAG [17][18][19] | 让模型学习“如何走图、何时回退、如何验证” | 潜力最大，但实现成本更高 |

从时间线上看：

- 2023 年的方法主要是“把 KG 事实喂给 LLM”或“让 LLM 作为图上 agent 做检索”。
- 2024 年开始，大量工作转向“计划-检索-推理”的解耦式框架，以及问题分解、逻辑形式生成、子图重排序。
- 2025 年的重点变成“faithfulness 与 relevance 同时优化”，出现了 FiDeLiS、Q-KGR、ORT、RJE、GNN-RAG、DoG 等更细粒度的方法。
- 截至 2026-04-01，我查到的最新工作大多是 2026 年 3 月的新 arXiv / 新接收论文，明显强调全局策略学习、可回退探索、路径验证和语义感知路径检索 [17][18][19]。

## 2. 近年主流方法的关键脉络

### 2.1 从“把 KG 贴到提示里”到“先过滤再注入”

早期路线的代表是 KAPING [2] 和 StructGPT [1]。这类方法默认 KG 的主要作用是给 LLM 提供额外事实，因此主要关注“如何把结构化知识转成 LLM 可读的输入”。它们的优点是实现简单、无需复杂训练；缺点是如果子图本身噪声较大，LLM 会被无关事实带偏。

2024-2025 年的改进重点不再是“把更多 KG 内容塞给 LLM”，而是“只把最相关的部分注入进去”。Q-KGR 明确指出，初始检索出的子图不可避免会包含干扰路径，因此先做 question-guided re-scoring，再做知识注入 [6]。AMAR 进一步把实体、关系、子图分成多个检索视角，并用 relevance gating 来决定哪些信息该保留、哪些该过滤 [8]。GNN-RAG 则用 GNN 在稠密子图上先做检索，再把最短路径作为 LLM 的证据上下文，核心优势是把问题相关性估计前移到了 LLM 之前 [12]。

这一条线对你当前问题最有启发的地方在于：**相关性控制最好发生在“进入 LLM 之前”或“每一步扩展之前”，而不只是生成结束后的后验筛选。**

### 2.2 从“直接生成答案”到“先生成计划 / 查询形式”

另一条非常重要的主线是“先把问题变清楚，再去图里找证据”。StructGPT [1] 通过 Iterative Reading-then-Reasoning 把 structured data 访问和语言推理解耦；ChatKBQA [5] 先生成逻辑形式，再检索并替换实体与关系；KELDaR [7] 更进一步，把复杂问题拆成 question decomposition tree，并对叶子级子问题做 atomic retrieval。

这类方法之所以对“路径合法但不相关”特别有效，是因为它们不是让模型直接在大图中盲走，而是先明确：

- 当前子问题到底在问什么；
- 下一步应该找哪类关系；
- 哪些路径虽然合法，但不满足当前子目标。

对于你的场景，这意味着一个非常现实的结论：**如果只靠前缀树约束当前 token 是否可生成，而没有显式的“子目标”或“路径意图”，那么 beam search 很容易偏向局部高频、局部通顺、但与问题目标不一致的路径。**

### 2.3 Agent 式图搜索：把“相关性”变成逐步决策问题

ToG [3] 和 RoG [4] 是这个方向的代表。ToG 把 LLM 当作在 KG 上做 beam search 的 agent，强调知识可追踪与可修正；RoG 用 planning-retrieval-reasoning 三阶段，把关系路径当作 faithful plan，再由计划去检索真实证据路径。两者的共同点是：**不是一次性把所有可能路径交给模型，而是逐步决定“下一步往哪里走”。**

2025 年之后，这条线开始更精细地建模“相关性是否足够”与“是否需要继续探索”：

- FiDeLiS 用 step-wise beam search 和 deductive scoring function，在每一步都检查当前证据是否足以推出答案，并引入 Path-RAG 缩小候选集 [11]。
- ORT 认为多跳 KGQA 的难点之一在于问题的目标语义很抽象，单靠从问题实体正向扩展往往找不到真正的目的节点，因此改为 ontology-guided reverse thinking，从“目的”反向构造推理路径 [13]。
- RJE 显式拆出 Retrieval-Judgment-Exploration 三个阶段：先取候选路径，再判断证据是否充分，只有不充分时才继续扩展 [14]。

这类工作最值得你关注，因为它们直接瞄准了当前痛点：**“合法扩展”不是“正确扩展”，中间还需要一个 judgment / verifier / scorer。**

### 2.4 图约束解码：faithfulness 很强，但 relevance 不一定够

GCR [9] 和 DoG [10] 代表了当前最强的“图结构直接进入生成过程”的路线。GCR 用 KG-Trie 限制 token 生成范围，从而得到完全 grounded 的 reasoning path；DoG 则强调生成 well-formed chains，用 graph-aware constrained decoding 保证链条从问题实体出发、沿 KG 合法延伸。

这一类方法的优点非常明确：

- 能显著减少 hallucination；
- 推理路径完全可追踪；
- 与图结构对齐得比普通 RAG 更紧。

但它们的局限也很典型：

- 约束通常是“局部可达性约束”，不是“全局问题相关性约束”；
- beam score 往往主要来自语言模型概率，容易偏向局部高频关系或更“语言上顺滑”的路径；
- 只要某条路径还在前缀树中，就可能被保留下来，即使它不是真正解决问题的路径。

换句话说，**约束解码更像是一个强 faithfulness filter，而不是一个强 relevance model。**

## 3. 为什么当前 GCR / KG-Trie 方案仍会“可达但不相关”

下面这几点是我结合你当前仓库的实现方式和相关文献做出的归纳，其中“问题相关性不足是系统性问题”这一点，和 Q-KGR、AMAR、ORT、RJE 的问题设定高度一致 [6][8][13][14]。

### 3.1 约束只检查“能不能走”，不检查“该不该走”

KG-Trie 约束的是前缀是否合法，因此它非常适合消除越界生成；但它不直接回答“这条边对当前问题是否有信息增益”。如果起点实体有许多语义上成立但问题无关的邻边，那么这些边都会进入候选空间。

### 3.2 beam search 会放大局部高频关系

如果当前模型是零样本、小模型、固定 hop 范围，并且生成分数主要受 token 概率控制，那么 beam search 很容易偏向：

- 出现频率更高的关系；
- 语言形式更稳定的路径模板；
- 图上分支更多、看起来更“有路可走”的节点。

这类偏差不会破坏 graph validity，但会破坏 question alignment。

### 3.3 固定 hop / 固定索引长度无法表达“目标意图”

从你当前脚本看，`index_path_length=2`，而且是 `zero-shot` + `beam` 搜索。这里我做一个明确推断：当问题的真实推理目标需要更强的中间语义约束时，固定 hop 的前缀树更像是在做“局部可达路径枚举”，而不是“目标导向规划”。这种设置对 faithfulness 友好，但对 relevance 不够敏感。

### 3.4 缺少“证据是否已足够”的停止准则

RJE [14] 和 FiDeLiS [11] 都强调一个关键点：系统不仅要会扩展，还要会判断“当前证据已经足够了吗”。如果没有这个判断模块，系统就会继续扩展许多仍然合法但已经偏题的路径。

## 4. 哪些文献思路最适合补到当前仓库

如果目标是尽量少改动当前 GCR 主体，而先解决“相关性漂移”，我认为最值得优先借鉴的是下面四条路线。

### 4.1 在 trie 约束之外，再加一个 question-conditioned path scorer

最直接的参考是 Q-KGR、AMAR、RJE [6][8][14]。做法不是取消 KG-Trie，而是把当前候选路径的分数从单一 `log P(path | prompt)` 改为混合打分：

```text
score(path) =
  lambda1 * llm_logprob
  + lambda2 * relevance(question, path)
  + lambda3 * support(answer | path)
  - lambda4 * redundancy(path)
```

其中 `relevance(question, path)` 可以先做一个低成本版本：

- 用 embedding / cross-encoder 对问题与路径文本化描述打分；
- 或让一个小模型只做 “relevant / irrelevant” 二分类；
- 或在每一步关系扩展前，对候选 relation 先排一次序。

这是最小改造、收益通常也最高的一步。

### 4.2 把“问题分解”接到路径生成前面

如果问题本身是复杂多跳问答，KELDaR 和 ChatKBQA 的启发很强 [5][7]。更合理的流程不是：

`问题 -> 直接在大图上做 beam`

而是：

`问题 -> 子目标/逻辑形式 -> 每个子目标各自约束扩展 -> 合并答案`

这样做的直接好处是：路径不再只需要“在图里存在”，还需要“满足当前子问题”。

### 4.3 引入 sufficiency / verifier 机制

FiDeLiS 和 RJE 说明，一个好的 KGQA 系统除了会找路径，还要会判断“这个路径是否已经足够支撑答案” [11][14]。对你当前管线，一个轻量可行的版本是：

- 对每条候选路径生成一个 “是否足以回答问题” 的判别分数；
- 如果分数低，则继续扩展；
- 如果分数高，则终止继续搜索并进入 answer aggregation。

这一步会显著减少“虽然还能往下走，但其实已经跑偏”的情况。

### 4.4 把正向扩展改成“目标导向扩展”

ORT [13] 和 S-Path-RAG [19] 给出的启发是：路径搜索不一定要完全从问题实体出发，也可以加入目标语义或答案类型语义。对当前仓库，这可以体现在：

- 先预测答案类型 / 目标概念；
- 用该目标语义对 relation / tail entity 做反向约束；
- 再把筛过的路径写入 trie 或用于 rerank。

这会比纯粹正向枚举更不容易跑偏。

## 5. 对当前项目的建议排序

如果以“改动小、见效快、还能保留当前 GCR 代码主体”为目标，我建议优先顺序如下：

1. **先加候选路径重排序器。**  
   不改解码器，只对已有 beam 结果做 question-conditioned rerank。这一步最容易复现，也最方便做 ablation。

2. **再加分步 relation gating。**  
   在每个 hop 扩展前先对 relation 打分，只保留 top-m 关系进入 trie 扩展。

3. **随后加入 sufficiency/verifier。**  
   明确判断什么时候停止，什么时候继续探索。

4. **最后再考虑训练式 agent / RL。**  
   如果前面三步已经验证“相关性问题”是主要瓶颈，再考虑 GraphWalker / KG-Hopper 这种成本更高但上限也更高的方案 [17][18]。

## 6. 总结

截至 2026-04-01，KG + LLM 推理问答领域最清晰的趋势不是继续增强“单纯的图约束”，而是把系统做成：

- 图结构保证合法性；
- 问题条件保证相关性；
- verifier 保证证据充分性；
- agent / policy 机制保证探索与回退能力。

因此，对当前仓库最合适的路线并不是“推翻 KG-Trie”，而是把它从“唯一约束”升级成“底层合法性层”，再在上层加上 relevance scorer、question decomposition 和 verifier。这样最符合现有文献趋势，也最贴近你现在遇到的“路径在 trie 中，但和问题无关”的具体问题。

## 参考文献

[1] Jiang et al. **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.574/

[2] Baek et al. **Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering**. 2023. https://arxiv.org/abs/2306.04136

[3] Sun et al. **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph**. ICLR 2024. https://arxiv.org/abs/2307.07697

[4] Luo et al. **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**. ICLR 2024. https://arxiv.org/abs/2310.01061

[5] Luo et al. **ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models**. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.122/

[6] Zhang et al. **Question-guided Knowledge Graph Re-scoring and Injection for Knowledge Graph Question Answering**. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.524/

[7] Li et al. **A Framework of Knowledge Graph-Enhanced Large Language Model Based on Question Decomposition and Atomic Retrieval**. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.670/

[8] Xu et al. **Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation**. AAAI 2025. https://arxiv.org/abs/2412.18537

[9] Luo et al. **Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models**. ICML 2025. https://proceedings.mlr.press/v267/luo25t.html

[10] Li et al. **Decoding on Graphs: Faithful and Sound Reasoning on Knowledge Graphs through Generation of Well-Formed Chains**. ACL 2025. https://aclanthology.org/2025.acl-long.1186/

[11] Sui et al. **FiDeLiS: Faithful Reasoning in Large Language Models for Knowledge Graph Question Answering**. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.436/

[12] Mavromatis and Karypis. **GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs**. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.856/

[13] Liu et al. **Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering**. ACL 2025. https://aclanthology.org/2025.acl-long.741/

[14] Lin et al. **RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs**. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.873/

[15] Sui et al. **Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering**. ACL 2025. https://aclanthology.org/2025.acl-long.622/

[16] Ma et al. **Large Language Models Meet Knowledge Graphs for Question Answering: Synthesis and Opportunities**. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1249/

[17] Xu et al. **GraphWalker: Agentic Knowledge Graph Question Answering via Synthetic Trajectory Curriculum**. 2026. https://arxiv.org/abs/2603.28533

[18] Wang and Yu. **KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning**. 2026. https://arxiv.org/abs/2603.21440

[19] Fu et al. **S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering**. 2026. https://arxiv.org/abs/2603.23512
