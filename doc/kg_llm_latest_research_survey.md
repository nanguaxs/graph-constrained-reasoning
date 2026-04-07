# KG 与 LLM 结合研究综述

更新日期：2026-04-01

本文独立于当前项目实现，单纯从研究全景出发，综述知识图谱（Knowledge Graph, KG）与大语言模型（Large Language Model, LLM）结合的最新进展，重点覆盖 2024-2026 年，并补充少量奠基工作。为了便于把握主线，本文沿用一个已经较为稳定的划分：`KG 增强 LLM`、`LLM 赋能 KG`、`KG 与 LLM 协同闭环系统` [1][2]。

## 摘要

截至 2026-04-01，KG 与 LLM 的结合已经明显从早期的“把三元组塞进提示词”发展为更完整的系统工程。研究重点正在从单纯的知识注入，转向以下几条主线：

- 用图结构改造检索增强生成，使 LLM 不只“看到事实”，还能“看到事实之间的关系” [7][8][9]。
- 用 KG 约束、验证或校正 LLM 推理过程，以提升 faithful reasoning、可解释性与抗幻觉能力 [5][6][12][13][14]。
- 反过来，让 LLM 参与 KG 构建、schema/ontology 生成、实体对齐、链接预测和图查询接口生成，推动知识工程流程自动化 [15][16][17][18][19][20][21][22][23][24][25][26][27][28]。
- 从“KG 是静态外部知识”进一步走向“KG 与 LLM 在运行中共同演化”，例如增量构图、实时个性化、查询执行与结果回写 [11][16][17][28]。

如果只用一句话概括这个领域过去两年的变化，那就是：

> KG 不再只是 LLM 的外挂记忆，而开始成为 LLM 检索、推理、验证、执行和知识维护的结构化操作系统。

## 1. 研究版图

| 主方向 | 典型问题 | 近年代表工作 | 2025-2026 的明显变化 |
| --- | --- | --- | --- |
| KG 增强 LLM | GraphRAG、可信推理、知识注入、事实校验、可解释推理 | StructGPT、MindMap、KG²RAG、GFM-RAG、InfuserKI、GCR [3][4][7][8][10][13] | 从静态提示走向可训练 GraphRAG、层级图检索、推理时验证 |
| LLM 赋能 KG | Text2KG、ontology/schema、entity alignment、KGC、Text2SPARQL/Cypher | iText2KG、AutoSchemaKG、LLMs4OL、LLMs4OM、LLM-Align、KG-LLM、Text2Cypher [16][17][20][21][22][23][26] | 从抽三元组走向 schema-first、增量构图、可执行查询与图数据库接口 |
| 协同闭环系统 | 增量知识更新、个性化、检索-执行-验证一体化 | Knowledge Graph Tuning、GraphRAFT、领域科学 KG 自动化 [11][19][28] | 从“读图”走向“建图、用图、改图”的循环系统 |

## 2. KG 增强 LLM

## 2.1 从结构化提示到图检索增强

较早的代表工作是 StructGPT 和 MindMap。StructGPT 的关键贡献不是简单把结构化知识转成文本，而是把 LLM 的工作拆成“读取结构化证据”和“基于证据推理”两步，显式区分外部接口访问与语言生成 [3]。MindMap 则把 KG 提示进一步组织为“graph of thoughts”，强调用图结构暴露潜在推理路径，同时改善可解释性与幻觉分析 [4]。

从 2024 年到 2026 年，GraphRAG 成为最突出的研究热点之一。KG²RAG 不再把检索上下文看成互不关联的独立 chunk，而是用 KG 提供 chunk 之间的事实关系，实现基于图的扩展与组织 [7]。GFM-RAG 则把图神经网络和 graph foundation model 引入 GraphRAG，使图结构本身能够学习 query-knowledge 关系，而不只是充当静态检索索引 [8]。到了 2026 年，TagRAG 进一步强调层级标签链、增量维护和检索效率，反映出 GraphRAG 已经开始从“研究 demo”走向“可维护系统” [9]。

这一条线最重要的变化是：KG 不再只是给 LLM 补充事实，而是在检索阶段显式组织证据之间的关系。也就是说，研究重点已经从“让 LLM 知道更多”转向“让 LLM 在结构化证据上工作”。

## 2.2 从知识注入到推理时校验

另一条主线是把 KG 作为知识注入或校验机制。InfuserKI 关注的是如何把新的 KG 知识高效注入到 LLM 中，同时降低知识遗忘，代表了训练时结构知识集成的一类思路 [10]。Knowledge Graph Tuning 更进一步，把 KG 用作实时个性化层：不改动 LLM 参数，而是从用户交互中抽取三元组，在线更新外部 KG，以实现可解释、低成本的个性化 [11]。

相比“训练时注入”，2024-2025 年更强的趋势其实是“推理时验证”。一类工作通过图检索增强来提升 factuality，另一类工作则直接把 KG 变成 LLM 推理的外部约束和验证器。Can Knowledge Graphs Make Large Language Models More Trustworthy? 这篇工作很重要，因为它不再默认“只要接上 KG 就会更可信”，而是系统评估了在开放式问答和污染图环境下，KG 对 LLM 可信性的真实影响 [12]。KG-LLM-Bench 也体现了类似取向，它强调的不只是准确率，而是文本化 KG 上的系统性推理能力与编码策略差异 [14]。

这说明一个越来越清晰的共识：KG 的价值不只在“提供知识”，更在“提供可以核验、可以追踪、可以被污染测试的知识边界”。

## 2.3 从“用图做上下文”到“在图上推理”

ToG、Reasoning on Graphs 和 GCR 代表了更偏“图上推理”的方向。Think-on-Graph 把 LLM 作为 agent，在 KG 上交互式探索实体与关系，并用 beam search 发现最有前途的路径 [5]。Reasoning on Graphs 则把问题求解拆成 planning-retrieval-reasoning 三个阶段，强调 faithful 和 interpretable 的图上推理 [6]。Graph-constrained Reasoning 则进一步把 KG 结构编码成生成约束，使 LLM 生成的 reasoning path 必须落在图上合法路径中 [13]。

这一方向的共同特征是：KG 不再只是输入上下文，而成为推理空间本身。相比传统 RAG，这类方法更接近 neuro-symbolic reasoning，因为它们显式考虑了路径、拓扑和结构可达性。

不过这条线目前也有明确张力：越强的图约束通常意味着越强的 faithfulness，但不一定自动带来更高的 relevance 或 usefulness。因此，2025-2026 的很多工作开始把图约束与 reranking、judgment、retriever learning 结合起来，而不是单独依赖路径合法性 [8][12][14]。

## 3. LLM 赋能 KG

## 3.1 Text2KG：从抽三元组到增量构图

LLM 对 KG 的反向赋能，最直接体现在 Text2KG。From human experts to machines 提出了一条更完整的流程：先由 LLM 辅助生成 competency questions，再构建 ontology/TBox，然后再基于文献生成和评估 KG，突出“知识工程流程自动化”而不是只做 relation extraction [15]。iText2KG 则把重点放在增量式构图上，目标是不依赖繁重后处理，就能在多种场景下持续整合实体、关系与图结构 [16]。

到了 2025 年，这个方向有两个特别明显的升级。第一，是 AutoSchemaKG 代表的 `schema-first` 趋势：不再默认先抽三元组、再事后补 schema，而是让 schema induction 成为构图的一部分 [17]。第二，是 Generating Domain-Specific Knowledge Graphs from Large Language Models 这类工作，它们开始尝试直接从 LLM 参数中“反向抽出”某一领域的结构知识，甚至能生成大规模领域 KG，但同时也暴露出 hallucination 会在自动扩图过程中逐步累积的问题 [18]。

到了 2026 年，Scientific knowledge graph and ontology generation using open large language models 进一步把这条路线推向科学场景，强调 open LLM 在长尾专业领域中自动生成 ontology 和 KG 的能力 [19]。这说明 Text2KG 已经从“能不能抽出来”变成了“能不能在缺少成熟 ontology 的领域里自动搭建知识骨架”。

## 3.2 Ontology、schema 和 entity alignment

如果说 Text2KG 解决的是“从文本中抽出知识”，那么 ontology learning、ontology matching 和 entity alignment 解决的是“让这些知识对齐到可复用的结构中”。LLMs4OL 是较早系统评估 LLM 用于 ontology learning 的工作，它把任务拆成 term typing、taxonomy discovery 和 non-taxonomic relation extraction，展示了 LLM 在零样本 ontology induction 上的潜力 [20]。LLMs4OM 则把焦点放到 ontology matching，说明 LLM 不只是能生成 ontology，还能参与异构 ontology 的对齐 [21]。

LLM-Align 把这一思路扩展到 entity alignment 场景，直接把 LLM 用作知识图谱对齐工具，以处理跨图实体匹配问题 [22]。从研究意义上看，这些工作很重要，因为它们把 LLM 从“文本生成器”变成了“语义整合器”。对未来的知识工程系统而言，自动 schema 生成、ontology 匹配和实体对齐几乎是必需能力。

## 3.3 LLM 用于 KG completion

另一类重要方向是让 LLM 直接参与知识图谱补全（KGC / link prediction）。KG-LLM for Link Prediction 是这条线的典型早期系统化尝试之一，它探索将 LLM 用作链接预测框架中的核心组件 [23]。Filter-then-Generate 则体现了 2025 年一个更成熟的趋势：不再单纯让 LLM 直接生成尾实体，而是先用结构过滤，再用结构-文本适配器把图表示与文本表示融合到一起 [24]。

这一方向说明了一个关键变化：KGC 研究开始从“语言模型能否直接补边”转向“如何把结构归纳偏置可靠地注入 LLM”。换句话说，未来更有前景的路线并不是拿 LLM 替代图模型，而是让 LLM 与结构建模模块协同。

## 3.4 自然语言到图查询

到 2025 年，KG 与 LLM 结合的另一个高增长点是自然语言图查询接口。Investigating Large Language Models for Text-to-SPARQL Generation 表明，即使不微调，仅靠 ICL 和多候选假设，LLM 也能在多个 KGQA 基准上生成具有竞争力的 SPARQL 查询 [25]。Text2Cypher 则把重心放在属性图和图数据库场景，强调高质量数据集和微调对 Cypher 生成性能的重要性 [26]。

SyntheT2C 代表了一个非常关键的趋势：当高质量 Text2Cypher 数据不足时，可以用合成数据反向训练 LLM 执行图查询生成 [27]。GraphRAFT 则进一步走向 retrieve-execute-reason 的闭环，它把图查询生成、图数据库执行和检索增强微调串起来，让 KG 不再只是 LLM 的“外部知识”，而是 LLM 的可执行后端 [28]。

这条线的意义在于，它让 KG 与 LLM 的结合从“知识增强”升级为“知识系统接口层”。一旦查询生成足够稳定，LLM 就可以自然地成为图数据库、企业知识平台和领域知识服务的统一入口。

## 4. KG 与 LLM 的协同闭环

所谓协同闭环，指的是 KG 与 LLM 不再只是单向增强，而是在运行过程中相互更新、相互校正。Knowledge Graph Tuning 已经体现了这一点：用户反馈不只是用来更新对话策略，而是被抽象成三元组进入 KG，再影响后续生成 [11]。AutoSchemaKG 与 iText2KG 也说明了类似趋势：KG 的 schema 和实例部分都可以在 LLM 驱动下持续演化 [16][17]。GraphRAFT 进一步把查询执行结果引入训练和推理流程，使“检索什么、执行什么、保留什么”成为统一流程的一部分 [28]。

这意味着一个更长远的研究方向已经出现：未来的 KG+LLM 系统不只是“LLM 调用外部图”，而是“LLM 和图共同组成一个持续演化的知识操作系统”。

## 5. 2025-2026 的关键趋势

### 5.1 GraphRAG 从静态文档图走向可训练、可维护图检索

KG²RAG 仍然更多是基于图组织检索结果，而 GFM-RAG 和 TagRAG 已经把训练、层级结构、增量维护和效率问题摆到了中心位置 [7][8][9]。这表明 GraphRAG 正在从“工程技巧”转向独立方法学。

### 5.2 Text2KG 从三元组抽取走向 schema-aware、ontology-aware

从 iText2KG 到 AutoSchemaKG，再到科学领域的 ontology + KG 共同生成，研究重点越来越集中在“先建立语义骨架，再扩充实例图” [16][17][19]。这比单纯抽三元组更接近真实知识工程需求。

### 5.3 研究重心从知识注入转向知识校验

InfuserKI 代表了训练时知识注入，而 OKGQA 和 KG-LLM-Bench 代表了对“知识是否可信、是否稳健、是否可污染测试”的关注 [10][12][14]。这说明社区越来越不满足于“看起来用了 KG”，而要求系统在可信性上给出可验证收益。

### 5.4 LLM 正在成为图系统的统一接口

Text-to-SPARQL、Text-to-Cypher、GraphRAFT 等工作共同说明，LLM 已经不只是被 KG 增强，它本身正在成为查询语言生成器、图数据库接口和知识服务入口 [25][26][27][28]。

### 5.5 从“读图”走向“建图、改图、用图”

KG+LLM 的研究对象正在扩张。早期工作多关注“如何利用图帮助推理”，而 2025-2026 的很多工作已经把自动构图、schema 诱导、alignment、个性化更新都纳入统一系统视角 [11][17][18][19][22]。

## 6. 仍然存在的核心挑战

### 6.1 图结构与语言空间之间仍有接口鸿沟

无论是 GraphRAG、KGC 还是 Text2Cypher，根本问题都没有变：图表示与语言表示属于不同归纳偏置。如何让 LLM 既保持语言灵活性，又真正使用结构约束，仍然是核心难题。

### 6.2 自动构图很容易放大 hallucination

Generating Domain-Specific Knowledge Graphs from Large Language Models 已经明确展示了这一点：自动生成规模越大，错误累积越明显 [18]。因此，未来高质量 Text2KG 不可能只靠生成，还需要 schema 约束、verification 和人机协作。

### 6.3 可信性评测还不够统一

OKGQA 和 KG-LLM-Bench 很重要，但它们仍主要聚焦问答和文本化 KG 推理 [12][14]。未来需要更系统地覆盖：

- 动态 KG
- 污染图 / 噪声图
- 多语言图
- 执行型任务
- schema 诱导与对齐任务

### 6.4 执行正确性比文本流畅性更难

Text-to-SPARQL 和 Text-to-Cypher 的研究一再说明：LLM 可以生成“像查询的文本”，但要生成“可执行且语义正确的查询”难得多 [25][26][27][28]。这也是为什么未来会越来越依赖 verifier、execution feedback 和 retrieval-then-execute 框架。

### 6.5 长尾领域知识工程仍离不开人

2026 年科学场景工作已经证明 open LLM 可以显著降低知识建模成本，但在专业长尾领域，ontology 与 KG 的高质量构建仍然高度依赖领域专家参与 [19]。短期内，更现实的方向是 human-in-the-loop，而不是 fully automatic。

## 7. 结论

KG 与 LLM 结合的研究已经从“象征知识增强生成”进入“结构化知识系统与生成模型共设计”的阶段。更准确地说，这个领域目前最有生命力的研究，不再是单点方法，而是围绕以下三个问题展开：

1. 如何让 LLM 在结构化证据上推理，而不是只把图当作额外文本。
2. 如何让 LLM 参与知识工程流程，而不是只消费现成知识。
3. 如何把检索、推理、查询执行、知识更新和验证整合成闭环系统。

因此，未来真正有潜力的 KG+LLM 系统，既不会是“纯图系统加一个聊天前端”，也不会是“纯 LLM 外挂一个三元组检索器”，而更可能是同时具备以下性质的混合体：

- 能读图；
- 能用图；
- 能查图；
- 能改图；
- 能解释为什么这样改。

## 参考文献

[1] Pan, J. Z., Razniewski, S., Kalo, J.-C., et al. **Unifying Large Language Models and Knowledge Graphs: A Roadmap**. 2023. https://arxiv.org/abs/2306.08302

[2] Pan, J. Z., Razniewski, S., Kalo, J.-C., et al. **Large Language Models and Knowledge Graphs: Opportunities and Challenges**. 2023. https://arxiv.org/abs/2308.06374

[3] Jiang, J., Zhou, K., Dong, Z., et al. **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**. EMNLP 2023. https://arxiv.org/abs/2305.09645

[4] Wen, Y., Wang, Z., Sun, J. **MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models**. 2023. https://arxiv.org/abs/2308.09729

[5] Sun, J., Xu, C., Tang, L., et al. **Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph**. ICLR 2024. https://arxiv.org/abs/2307.07697

[6] Luo, L., Zhao, Z., Gong, C., et al. **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**. ICLR 2024. https://arxiv.org/abs/2310.01061

[7] Zhu, X., Xie, Y., Liu, Y., Li, Y., Hu, W. **Knowledge Graph-Guided Retrieval Augmented Generation**. NAACL 2025. https://aclanthology.org/2025.naacl-long.449/

[8] Luo, L., Zhao, Z., Haffari, G., et al. **GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation**. NeurIPS 2025. https://arxiv.org/abs/2502.01113

[9] Tao, W., Li, X., Lan, Y., Qian, W. **TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation**. 2026. https://arxiv.org/abs/2601.05254

[10] Wang, F., Bao, R., Wang, S., et al. **InfuserKI: Enhancing Large Language Models with Knowledge Graphs via Infuser-Guided Knowledge Integration**. Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.209/

[11] Sun, J., Du, Z., Chen, Y. **Knowledge Graph Tuning: Real-time Large Language Model Personalization based on Human Feedback**. 2024. https://arxiv.org/abs/2405.19686

[12] Sui, Y., Wang, Y., Wang, Y., et al. **Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering**. ACL 2025. https://aclanthology.org/2025.acl-long.622/

[13] Luo, L., Zhao, Z., Gong, C., et al. **Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models**. ICML 2025. https://proceedings.mlr.press/v267/luo25t.html

[14] Besta, M., Chhabra, A., Gerstenberger, R., et al. **KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs**. 2025. https://arxiv.org/abs/2504.07087

[15] Kommineni, V. K., König-Ries, B., Samuel, S. **From human experts to machines: An LLM supported approach to ontology and knowledge graph construction**. 2024. https://arxiv.org/abs/2403.08345

[16] Lairgi, Y., Moncla, L., Cazabet, R., et al. **iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models**. 2024. https://arxiv.org/abs/2409.03284

[17] Xu, X., Yu, H., Li, X., et al. **AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora**. 2025. https://arxiv.org/abs/2505.23628

[18] Parović, M., Li, Z., Du, J. **Generating Domain-Specific Knowledge Graphs from Large Language Models**. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.602/

[19] Oarga, A., Hart, M., Bran, A. M., et al. **Scientific knowledge graph and ontology generation using open large language models**. Digital Discovery, 2026. https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00275c

[20] Babaei Giglou, H., D'Souza, J., Auer, S. **LLMs4OL: Large Language Models for Ontology Learning**. 2023. https://arxiv.org/abs/2307.16648

[21] Babaei Giglou, H., D'Souza, J., Engel, F., Auer, S. **LLMs4OM: Matching Ontologies with Large Language Models**. 2024. https://arxiv.org/abs/2404.10317

[22] Chen, Y., Hao, R., Zhang, C., et al. **LLM-Align: Utilizing Large Language Models for Entity Alignment in Knowledge Graphs**. 2024. https://arxiv.org/abs/2412.04690

[23] Nyilas, M., Schlegel, V., Berrendorf, M., et al. **Knowledge Graph Large Language Model (KG-LLM) for Link Prediction**. 2024. https://arxiv.org/abs/2403.07311

[24] Zhang, S., Sun, D., Yang, T., et al. **Filter-then-Generate: Large Language Models with Structure-Text Adapter for Knowledge Graph Completion**. 2024. https://arxiv.org/abs/2412.09094

[25] D’Abramo, J., Zugarini, A., Torroni, P. **Investigating Large Language Models for Text-to-SPARQL Generation**. KnowledgeNLP 2025. https://aclanthology.org/2025.knowledgenlp-1.5/

[26] Ozsoy, M. G., Messallem, L., Besga, J., Minneci, G. **Text2Cypher: Bridging Natural Language and Graph Databases**. GenAIK 2025. https://aclanthology.org/2025.genaik-1.11/

[27] Jovanović, J., Neumaier, S., Breit, A. **SyntheT2C: Generating Synthetic Data for Fine-Tuning Large Language Models on the Text2Cypher Task**. 2024. https://arxiv.org/abs/2406.10710

[28] Xie, S., Feng, J., Yap, G. E., et al. **GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases**. 2025. https://arxiv.org/abs/2504.05478
