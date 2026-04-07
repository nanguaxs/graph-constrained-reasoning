# ToG 与 GCR 融合方案

更新日期：2026-04-02

本文总结三种可行的 ToG 与 GCR 融合方案，并结合当前仓库代码解释为什么这些方案可落地。同时，专门解释“方案三”到底在做什么，以及当前代码中的 `add_rule` 实际承担了什么作用。

## 1. 先给一个总判断

ToG 和 GCR 不是互斥关系，而是天然互补：

- ToG 擅长决定“往哪走”，核心是问题相关性、逐步搜索和证据充分性判断。
- GCR 擅长决定“怎么走得合法”，核心是图约束解码、faithfulness 和路径可追踪。

因此，一个很自然的融合思路是：

> 用 ToG 控制搜索方向，用 GCR 保证搜索结果始终落在知识图谱上。

从当前仓库代码看，系统已经有比较完整的 GCR 底座：

- 在 `src/qa_prompt_builder.py` 中构建路径索引与 trie；
- 在 `src/graph_constrained_decoding.py` 中做 token 级图约束；
- 在 `workflow/predict_paths_and_answers.py` 中完成约束路径生成；
- 在 `workflow/predict_final_answer.py` 中基于路径再回答问题。

但目前还缺少 ToG 式的“每一步 relation/entity 判断”和“是否继续探索”的控制层。

## 2. 当前 GCR 主流程在代码里是怎样的

### 2.1 路径索引的构建

在 `src/qa_prompt_builder.py` 中，`GraphConstrainedPromptBuilder.get_graph_index()` 和 `JointReasoningPromptBuilder.get_graph_index()` 会先从图里拿出候选路径：

- 默认情况：直接 `dfs(g, question_dict["q_entity"], self.index_path_length)`；
- 如果开启 `add_rule`：优先走 `apply_rules()`，只有规则为空时才回退到 `dfs()`。

也就是说，当前默认路径空间通常是：

`问题实体出发 -> 固定 hop 的 DFS 路径全集`

### 2.2 路径生成

在 `workflow/predict_paths_and_answers.py` 中，`prediction()` 会先拿到：

- `input_query`
- `ground_paths`
- `trie`

然后调用 `model.generate_sentence(...)`，把 trie 交给 GCR 解码器。

### 2.3 图约束解码

在 `src/graph_constrained_decoding.py` 中，`allowed_tokens_fn()` 会在 `<PATH>` 到 `</PATH>` 之间调用 `self.trie.get(...)`，只允许模型生成当前 trie 前缀下合法的下一个 token。

这一步保证了：

- 模型不会生成图上不存在的路径；
- 生成路径是 grounded 的；
- 可解释性和可追踪性较强。

但这一步并不判断：

- 这条路径和问题是否真的相关；
- 是否应该继续往这个分支扩；
- 是否已经找到足够的答案。

这正是 ToG 能补的地方。

## 3. 三种融合方案

## 3.1 方案一：ToG 前置筛选，GCR 约束生成

### 核心思路

先用 ToG 风格的方法筛掉明显无关的 relation / entity / path，再把剩下的候选写进 trie，最后仍然用 GCR 来生成路径。

流程如下：

`问题 -> 候选 relation/entity 打分 -> 过滤子图 -> 构建更小的 trie -> GCR 约束生成路径 -> 最终回答`

### 适合怎么改代码

最适合插入的位置是 `src/qa_prompt_builder.py` 中的 `get_graph_index()`：

- 当前做法是直接 `dfs()` 枚举固定 hop 路径；
- 可以改成先生成候选 relation，再按问题打分，最后只保留 top-m relation 对应路径。

### 优点

- 改动最小；
- 不需要动 `GraphConstrainedDecoding`；
- 仍然保留 GCR 的完整路径合法性；
- 最适合先做 ablation。

### 缺点

- 仍然是“一次性建好大部分路径空间”，不是像 ToG 一样逐步决策；
- 对多答案问题的覆盖控制还不够强。

## 3.2 方案二：ToG 逐步搜索，GCR 每一步局部约束

### 核心思路

不再一开始把整条 2-hop/3-hop 路径空间都建好，而是像 ToG 那样逐 hop 决策。每一跳先判断“下一步最可能该走哪些关系”，再用 GCR 只在这一小块局部候选上做约束生成。

流程如下：

`问题 -> 第 1 hop 关系筛选 -> 局部 trie 约束生成 -> frontier 更新 -> 第 2 hop 再筛选 -> 再生成 -> ... -> 判断是否停止`

### 适合怎么改代码

最适合改造的入口是：

- `workflow/predict_paths_and_answers_sequential_sampling.py`

因为这个脚本已经有：

- 多轮生成；
- 去重；
- blocked trie 重构；
- 逐轮补采样。

这些机制天然适合扩展成：

- frontier 搜索；
- relation gating；
- sufficiency 判断；
- coverage-aware stopping。

### 优点

- 最接近 ToG 原论文风格；
- 问题相关性会明显更强；
- 更适合多答案和多路径覆盖；
- 更容易加“什么时候停止”的判断。

### 缺点

- 代码改动最大；
- 工程复杂度高；
- 对搜索状态管理要求更高。

## 3.3 方案三：ToG 只生成“关系规则/关系计划”，GCR 负责执行

### 核心思路

ToG 不直接负责找完整路径，而只是先判断“应该沿哪些关系模式去找”。然后把这些关系模式交给 GCR，让 GCR 在真实图里执行这些模式，枚举出所有满足模式的 grounded paths。

流程如下：

`问题 -> ToG 生成 relation rule / relation plan -> 用 rule 在图中搜满足规则的路径 -> 用这些路径构建 trie -> GCR 约束生成 grounded paths -> 回答`

### 这是最贴当前代码的一种

原因是你现在代码里已经存在这条链：

- `--add_rule`
- `merge_rule_result(...)`
- `question_dict["predicted_paths"]`
- `apply_rules(...)`
- `utils.bfs_with_rule(...)`

也就是说，这个仓库本来就支持：

> 先给一组“规则”，再在图里执行这些规则，从而缩小 GCR 的候选路径空间。

所以方案三不是空想，而是最容易接进现有代码的融合方式。

## 4. 方案三到底是什么意思

你刚才说“没太看懂”，主要是因为这里的“rule”容易让人误以为是完整路径，其实它更像“关系模板”。

### 4.1 用一句话解释

方案三的本质是：

> ToG 先告诉 GCR “往哪类关系上找”，GCR 再去图里把真正的实体路径找出来。

### 4.2 一个直观例子

假设问题是：

`谁是某某省所在国家使用的货币？`

ToG 不一定要直接给出完整路径：

`某省 -> 所属国家 -> 国家 -> 使用货币 -> 货币`

它也可以只先给出关系计划：

- 第一步优先找 `located_in_country`
- 第二步优先找 `currency_used`

这时 GCR 负责做的事情是：

- 从问题实体出发；
- 在 KG 中找所有满足这个关系序列的真实路径；
- 把这些真实路径写进 trie；
- 然后模型只能在这些 grounded paths 里生成。

因此：

- ToG 决定“关系方向”
- GCR 决定“真实落图”

### 4.3 为什么这样有意义

因为 ToG 的优势在于“问题理解与搜索意图”，而 GCR 的优势在于“路径必须合法”。

如果直接让 ToG 输出完整路径，会有两个风险：

- 可能仍然出现语言幻觉；
- 输出的实体链不一定都真能在图里对应上。

而如果只让 ToG 先给“关系规则”，就会稳很多：

- 规则空间比完整路径空间小；
- 更容易让模型判断“下一步关系该是什么”；
- 最终实体实例化交给图搜索去做，不容易飘。

### 4.4 所以方案三不是“只用规则回答”

这一点很重要。

方案三不是：

`问题 -> relation rule -> 直接输出答案`

而是：

`问题 -> relation rule -> 在 KG 里执行 rule -> 得到 grounded paths -> 再回答`

所以它本质上是“规则引导的 GCR”，不是“规则替代 GCR”。

## 5. 当前代码里的 add_rule 有什么用

一句话总结：

> `add_rule` 的作用是把一组外部给定的规则/路径预测结果先注入进来，用这些规则在图中筛出更小、更定向的候选路径集合，而不是直接对整张图做无差别 DFS。 

## 5.1 在路径生成阶段，add_rule 用来缩小 trie 的候选空间

相关代码：

- `workflow/predict_paths_and_answers.py`
- `src/qa_prompt_builder.py`
- `src/utils/graph_utils.py`

### 具体链路

#### 第一步：把规则文件合并进数据集

在 `workflow/predict_paths_and_answers.py` 里：

- 开启 `--add_rule` 后，会从 `rule_path` 读取一个 jsonl；
- `merge_rule_result()` 把每个问题对应的 `prediction` 写进样本的 `predicted_paths` 字段。

这意味着：

- 原始 QA 样本本来只有 `question / graph / q_entity / a_entity`
- 合并后多了一个 `predicted_paths`

#### 第二步：在 prompt builder 中根据规则执行图搜索

在 `src/qa_prompt_builder.py` 中：

- 如果 `self.add_rule == True`
- 就读取 `question_dict["predicted_paths"]`
- 然后调用 `apply_rules(...)`

而 `apply_rules(...)` 又会对每个起点实体、每条规则调用：

- `utils.bfs_with_rule(graph, entity, rule)`

#### 第三步：bfs_with_rule 做的其实是“按关系序列找真实路径”

在 `src/utils/graph_utils.py` 中，`bfs_with_rule(graph, start_node, target_rule)` 的逻辑是：

- `target_rule` 是一个关系序列；
- 从 `start_node` 出发做 BFS；
- 只有当前边的 relation 和规则中当前位置匹配，才继续走；
- 最后得到所有满足这条关系序列的真实路径。

所以 `add_rule` 的作用不是“直接把规则塞给模型”，而是：

> 先拿规则在 KG 中找一批真实路径，再把这些真实路径转成 trie，最后才交给 GCR 做受约束生成。

### 这意味着什么

如果规则质量好：

- 候选路径空间会小很多；
- 与问题相关的路径比例会更高；
- 生成更稳定。

如果规则质量差：

- 可能把真正有用的路径都剪掉；
- 也可能因为规则为空而回退到 DFS。

## 5.2 在最终答题阶段，add_rule 用来构造“规则对应的推理路径上下文”

相关代码：

- `workflow/predict_final_answer.py`
- `src/qa_prompt_builder.py` 中的 `PromptBuilder`

在 `predict_final_answer.py` 中开启 `--add_rule` 后，同样会先把规则合并进数据集。  
然后 `PromptBuilder.process_input()` 在 `self.add_rule=True` 时会：

- 读取 `predicted_paths`
- 调用 `apply_rules(...)`
- 得到满足规则的真实 reasoning paths
- 把这些路径转成文本，拼进 prompt 的 `Reasoning Paths:` 上下文里

也就是说，在 final answer 阶段：

- `add_rule` 不是直接提供答案；
- 它提供的是“由规则展开得到的路径证据”。

这一步和路径生成阶段其实是一致的，只是用途不同：

- 路径生成阶段：它帮助构建更小的 trie
- 最终答题阶段：它帮助构建更相关的上下文

## 5.3 当前 add_rule 更像“规则过滤器”，还不是完整的 ToG

这也是最关键的判断。

当前 `add_rule` 已经具备了方案三的雏形，但还不等于真正的 ToG-GCR 融合。因为它现在缺的主要是：

- 没有问题条件化的 relation scoring；
- 没有逐 hop 的动态决策；
- 没有 sufficiency 判断；
- 没有基于答案覆盖率的停止逻辑。

所以你可以把当前 `add_rule` 理解成：

> 一个已经存在的“规则注入接口”，非常适合拿来承载方案三。

## 6. 三个方案怎么选

如果按“改动小、收益快、最贴当前代码”来排，我建议顺序是：

1. **先做方案三**
   原因是 `add_rule` 这条链已经存在，你只需要把 rule 的来源从“外部已有规则文件”升级成“ToG-style relation planner 输出”。

2. **再做方案一**
   当你有了更稳的 rule 生成后，再把规则扩展到 relation/entity 级过滤，进一步缩小 trie。

3. **最后做方案二**
   等前两步跑通，再升级到逐 hop 搜索和动态停止，这是最强但也最复杂的版本。

## 7. 一个最推荐的落地路线

我最建议你先做下面这个最小实现：

### 第一步

先做一个 ToG-lite 模块，只输出：

- top-m relation
- 或 top-m relation sequence

### 第二步

把这些 relation sequence 转成当前代码能直接吃的 `predicted_paths` / `rules` 格式。

### 第三步

直接复用现有 `--add_rule` 机制：

- `merge_rule_result(...)`
- `apply_rules(...)`
- `bfs_with_rule(...)`
- `get_graph_index(...)`

### 第四步

仍然使用 GCR 生成 grounded paths，再回答问题。

这样做的最大好处是：

- 不用先重写整个解码器；
- 不用一上来就改成逐 hop agent；
- 可以快速验证“问题相关性是否显著提高”。

## 8. 最后一句话总结

如果你只记住一句话，我建议记这句：

> 方案三其实就是“让 ToG 先给搜索意图，让 GCR 负责把这个意图落实到图上的真实路径”。

而当前代码里的 `add_rule`，恰好就是承载这个思路的现成接口。

