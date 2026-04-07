# GCR 改造建议

更新日期：2026-04-01

本文记录针对当前仓库 `graph-constrained-reasoning` 的具体改造建议，目标是解决一个核心问题：

> 当前 KG-Trie / 前缀树约束解码可以保证生成路径在知识图谱上合法，但有时生成的路径虽然在图中可达，却和问题本身无关。

这份文档不再重复做文献综述，而是把建议直接落到当前代码结构上，回答三个问题：

1. 当前系统的瓶颈具体在哪里。
2. 哪些改动最值得优先做。
3. 每条建议应该接到现有代码的哪一层。

## 1. 当前管线的工作方式

从代码实现看，当前主流程大致如下：

1. 在 `workflow/predict_paths_and_answers.py` 中读取样本。
2. 由 `src/qa_prompt_builder.py` 中的 `ChinesePathGenerationWithAnswerPromptBuilder` 构造 prompt 和 KG-Trie。
3. `get_graph_index()` 从问题实体出发，对图做固定深度 DFS，枚举长度受限的路径，并全部放入 `MarisaTrie`。
4. `src/llms/graph_constrained_decoding_model.py` 调用 Hugging Face `generate()`。
5. `src/graph_constrained_decoding.py` 中的 `prefix_allowed_tokens_fn` 在解码时按 trie 约束后续 token。
6. 最终把 beam 或 group-beam 生成结果直接写入 `predictions.jsonl`。

用一句话总结当前系统：

`问题实体附近所有合法路径 -> 受 trie 约束的 beam search -> 直接输出`

这个流程在“合法性”上很强，但在“相关性”上几乎没有独立建模层。

## 2. 为什么会出现“合法但不相关”

### 2.1 候选空间是“可达空间”，不是“目标空间”

当前 trie 的构造方式本质上是：

- 从问题实体出发；
- 按固定 hop 枚举路径；
- 只要路径在图中存在，就进入候选集合。

这意味着 trie 保证的是：

- 路径合法；
- 路径可达；
- 路径符合图结构。

但它并不保证：

- 路径与问题语义匹配；
- 路径能提供回答问题所需证据；
- 路径的终点实体类型符合答案类型。

因此，只要起点实体邻边很多，候选空间里就必然包含大量“合法但无关”的路径。

### 2.2 当前排序主要依赖语言模型分数

虽然生成被 trie 约束了，但 beam search 本身仍然主要依赖语言模型概率。这样会天然偏向：

- 高频关系；
- 更顺滑、更模板化的路径表达；
- 分支多、延展性强的节点。

这些偏好未必和“回答当前问题”一致。

### 2.3 固定 hop 会放大噪声

当前脚本里使用了固定的 `index_path_length=2`。这种设计有两个典型问题：

- 对简单问题来说，2-hop 可能过长，带来额外噪声。
- 对复杂问题来说，2-hop 可能不够，导致真正相关路径没有被完整表达出来。

于是系统容易退化成“固定深度的局部路径枚举器”，而不是“面向问题目标的推理器”。

### 2.4 当前没有“证据是否足够”的判断层

系统现在的停止条件接近于：

- 路径生成到 `</PATH>` 就结束；
- 或达到生成器本身的停止条件。

但并没有一层显式判断：

- 这条路径是否足以支持答案；
- 是否应该继续扩展；
- 是否应该提前停止并交给答案模块。

缺少这一层，系统就容易保留很多“还能继续走，但已经偏题”的路径。

## 3. 总体改造方向

最核心的原则不是推翻 KG-Trie，而是把它从“唯一约束”升级成“底层合法性层”。

建议把系统逐步改造成三层：

1. `合法性层`：继续由 KG-Trie 保证路径在图上成立。
2. `相关性层`：判断候选路径是否与当前问题相关。
3. `充分性层`：判断当前证据是否已经足以回答问题。

也就是说，未来理想流程应当更接近：

`问题 -> 候选路径生成（合法） -> 路径打分/筛选（相关） -> 证据判断（充分） -> 输出`

## 4. 具体建议

## 4.1 建议一：先加 post-hoc reranker

这是最推荐优先做的改动。

### 目标

不要让 beam 的原始顺序直接决定最终结果，而是在已有候选路径上做一次“问题相关性重排序”。

### 为什么最值得先做

- 不需要改动 KG-Trie 的主体逻辑。
- 不需要改写底层 constrained decoding。
- 可以直接利用当前 `beam` / `group-beam` 已经返回的 `k` 条候选。
- 非常适合做 ablation，能最快验证“相关性是否真是主要瓶颈”。

### 建议接入位置

接在 `workflow/predict_paths_and_answers.py` 中调用 `model.generate_sentence(...)` 之后。

当前逻辑大致是：

```text
input_builder.process_input(...)
-> model.generate_sentence(...)
-> 直接写 prediction
```

建议改成：

```text
input_builder.process_input(...)
-> model.generate_sentence(...)
-> rerank(prediction_candidates, question)
-> 写 top-1 或 top-n
```

### 最小实现版本

先不做复杂训练，直接做一个轻量 scorer：

- 输入：`question + path_text`
- 输出：一个 relevance 分数

可以尝试的实现方式：

1. embedding 相似度
2. cross-encoder 打分
3. 小模型二分类：`relevant / irrelevant`

### 更理想的打分函数

```text
final_score(path) =
  a * beam_score
  + b * relevance(question, path)
  + c * answer_type_match(path)
  - d * redundancy(path)
```

其中：

- `beam_score` 表示原始生成分数；
- `relevance(question, path)` 表示问题与路径的语义匹配程度；
- `answer_type_match(path)` 表示路径末端实体类型是否符合答案类型；
- `redundancy(path)` 用来惩罚重复或模板化候选。

### 训练数据如何构造

如果你数据里有 gold path，可以很自然地构造 reranker 训练集：

- 正样本：ground truth path
- 难负样本：同一问题下 beam 生成但错误的路径
- 额外负样本：同一起点实体出发、语义接近但答案错误的路径

这类 hard negative 往往对“合法但不相关”的错误特别有效。

## 4.2 建议二：在构 trie 前先做 relation gating

如果 rerank 是“生成后再挑”，那 relation gating 就是“生成前先少放错的候选进去”。

### 目标

在固定 hop 的 DFS 之前，先根据问题筛掉明显无关的关系或三元组。

### 当前问题

现在 `get_graph_index()` 会把从问题实体出发的所有 DFS 路径都放入 trie。  
这样 trie 的覆盖面很大，但噪声也很大。

### 建议做法

把流程改成：

`问题 -> 预测相关 relation / triple -> 只保留 top-m -> 再构 trie`

### 为什么这一步很重要

- 它直接减少“合法但无关”的路径进入候选空间。
- 它比后验 rerank 更早地把问题语义注入搜索空间。
- 它能减小 beam search 的负担。

### 与当前仓库的契合点

`src/qa_prompt_builder.py` 中已经存在 `RetrievalPromptBuilder`，而且已经支持：

- entity trie
- relation trie
- triple trie

这意味着你其实已经有了一部分“按结构元素做检索/约束”的基础设施。  
非常适合拿来扩展 relation gating，而不是从零新写一套。

### 两种推荐实现

轻量版：

- 只对一跳 relation 做打分；
- 只保留 top-m relation；
- 再 DFS 扩展。

增强版：

- 在每一步 hop 扩展时都动态地给候选 relation 打分；
- 形成 step-wise relation gating。

如果想先快速验证，我建议先做轻量版。

## 4.3 建议三：加入 verifier / sufficiency judge

这一步的目标不是找更多路径，而是明确判断：

> 当前路径是否已经足以支持回答问题？

### 为什么需要这一步

当前系统默认“生成出一条合法路径”就足够了，但实际上很多路径只是相关，不一定充分；也有很多路径在表面上合理，但缺少真正决定答案的证据。

### 建议接入位置

可以有两种接法：

1. 在 beam 完成后，对每条路径增加一个 `supportive / partially_supportive / irrelevant` 判断。
2. 更进一步，在每个 hop 后判断是否还需要继续扩展。

### 轻量可行方案

先做第一种：

- 对当前候选路径生成一个“是否足以回答问题”的分数；
- 分数低则过滤；
- 分数高则保留到最终答案候选。

### 更强的升级方向

把固定深度搜索改成动态深度：

`1-hop -> verifier -> 不够再扩展 -> verifier -> 直到足够或达到上限`

这能明显降低固定 `index_path_length` 带来的噪声问题。

## 4.4 建议四：加入答案类型约束或目标语义约束

这一步适合在前两步有效后继续增强。

### 核心思想

问题不仅包含起点实体，还包含“目标类型”信息，例如：

- 问的是谁；
- 问的是哪家公司；
- 问的是哪个地点；
- 问的是哪个时间点。

这些信息可以作为路径筛选的重要线索。

### 建议做法

先预测：

- 答案类型；
- 目标概念；
- 可能的目标关系；

再用这些信息限制：

- relation 候选；
- tail entity 类型；
- path rerank 分数。

### 为什么有效

很多“合法但不相关”的路径，本质上不是图结构错了，而是终点类型就不对。  
如果问题在问“人”，而路径终点更像“地点”或“组织”，它即使在图上合法，也不应该被优先保留。

## 4.5 建议五：长期方向是 agent 化的逐步图搜索

这不是最适合第一步做的，但如果前面几步验证有效，后续可以走到更完整的 `relevance-aware graph search`。

### 与当前方案的区别

当前系统更像：

- 先把所有可达路径放好；
- 再让模型在其中生成。

而 agent 化图搜索更像：

- 当前看到什么；
- 判断下一跳往哪里走；
- 发现不对就回退；
- 直到证据充分。

### 为什么不建议一开始就做

- 工程改动大；
- 很难快速确认收益来源；
- 不利于做干净的 ablation。

所以更合理的顺序是先做 `rerank -> gating -> verifier`，最后再决定是否值得上 agent 化方案。

## 5. 一个必须先注意的实现风险

这是一个很重要的实现细节，和论文思路无关，但会影响实验结论。

在当前 `src/graph_constrained_decoding.py` 的逻辑里，如果当前前缀在 trie 中查不到允许的后继 token，会回退到“允许所有 token”。

这意味着：

- 约束并不是始终严格生效；
- 一旦发生 prefix mismatch，系统会暂时退回无约束生成；
- 后续结果里可能混入“相关性问题”和“约束泄漏问题”。

建议至少做一项改动：

1. 把 fallback 改成更保守的结束策略；
2. 或直接判当前 beam 无效；
3. 至少把 fallback 次数记录到日志里。

如果不处理这件事，后面做 rerank/gating 的实验时，很难判断收益到底来自哪里。

## 6. 推荐的实施顺序

如果目标是“尽量少改代码、尽快得到可验证收益”，建议按下面顺序推进：

1. 先补日志和诊断信息
2. 再做 post-hoc reranker
3. 再做 relation gating
4. 再做 verifier / sufficiency judge
5. 最后再考虑问题分解或 agent 化图搜索

## 7. 推荐先做的最小可行版本

如果只选一个最值得先落地的版本，我推荐：

`KG-Trie + beam/group-beam + post-hoc reranker + 路径支持度判断`

理由是：

- 保留当前主框架；
- 不需要立刻重写解码器；
- 最容易解释实验收益；
- 最贴合当前“合法但不相关”的实际问题。

## 8. 推荐的实验设计

为了避免改动太多后无法定位收益来源，建议按下面方式做 ablation：

### Base

- 原始 GCR / KG-Trie
- `beam k=8`
- `index_path_length=2`

### Exp-1

- Base + reranker

看 reranker 是否能显著提升：

- top-1 path relevance
- answer accuracy
- ground truth path hit

### Exp-2

- Base + relation gating

看 relation gating 是否能：

- 降低无关路径比例；
- 缩小候选空间；
- 提升生成稳定性。

### Exp-3

- Base + reranker + relation gating

验证两者是否互补。

### Exp-4

- Base + reranker + relation gating + verifier

验证“证据充分性”这一层是否继续带来提升。

## 9. 总结

当前系统最大的优点是：

- 图约束做得很强；
- faithfulness 很好；
- 路径可追踪性很高。

当前系统最大的短板是：

- 缺少独立的相关性建模层；
- 缺少证据充分性判断层；
- 固定 hop 搜索容易放大噪声。

因此，最合理的方向不是放弃 KG-Trie，而是补齐它上面的两层：

- `relevance scorer`
- `sufficiency verifier`

如果这两层验证有效，再进一步考虑 relation gating、问题分解和 agent 化图搜索，会更稳、更容易解释。
