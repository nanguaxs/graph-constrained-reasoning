# SFT + GRPO 效果分析

更新日期：2026-04-01

本文基于 `finetune/` 目录中的实现，分析为什么在这个项目里采用“先 SFT，再 GRPO”的两阶段微调后，效果会比较好。这里的“效果不错”我理解为：模型更容易生成合法、接近真值、终点更正确、结构更干净的路径。

有一个前提假设需要先说明：

- 你实际跑实验时，GRPO 阶段不是重新从原始基座模型开始，而是接在 SFT 之后继续训练，或者至少把 `GRPOConfig.model_name` 指向了 SFT 后导出的模型/合并模型。

如果这个假设成立，那么当前实现下的训练过程并不是“普通 SFT + 普通 RLHF/GRPO”，而更接近：

> 一个已经学会路径格式与基本路径分布的模型，在图约束的候选空间里，接受带教师锚点和密集奖励的、相对保守的策略优化。

这正是它稳定且有效的关键。

## 1. 训练链路复原

## 1.1 SFT 在做什么

SFT 的目标非常直接，不是让模型泛化地“回答问题”，而是让模型在当前 prompt 形式下输出 GT path。

从代码看：

- `SFTPathDataset` 会把每一条 ground-truth path 展开成一个独立监督样本，见 `finetune/dataset.py:156-176`。
- prompt 文本复用了 `ChinesePathGenerationWithAnswerPromptBuilder`，因此训练输入格式和主推理流程高度一致，见 `finetune/dataset.py:90-96`。
- 训练时只监督路径部分，prompt token 的 label 全部置为 `-100`，见 `finetune/dataset.py:216-217`。
- `<PATH>` / `</PATH>` 特殊 token 在 SFT 开始前被显式加入 tokenizer，并同步扩展 embedding，见 `finetune/sft_train.py:71-74`。

因此，SFT 学到的东西很“干净”：

- 学会在正确的位置开始生成路径；
- 学会路径的序列格式；
- 学会常见实体-关系-实体串联模式；
- 学会从当前 prompt 中抽取与路径生成最相关的信息。

## 1.2 GRPO 在做什么

GRPO 阶段并不是在一个完全开放的 action space 上做自由探索，而是在一个非常受控的图约束空间里做相对优化。

从代码看：

- rollout 时继续使用 `GraphConstrainedDecoding`，并把约束默认打开，见 `finetune/grpo_trainer.py:260-264`。
- 每个样本先生成一组路径，再按 reward 做组内标准化 advantage，见 `finetune/grpo_trainer.py:420-435` 与 `finetune/reward.py:358-370`。
- rollout 不是简单重复采样，而是做了去重、路径 token 屏蔽和多轮补生成，见 `finetune/grpo_trainer.py:292-398`。
- 更重要的是，训练组里会先注入最多 1-2 条 GT path，再去补生成其他路径，见 `finetune/grpo_trainer.py:205-225` 与 `finetune/grpo_trainer.py:303-320`。

这意味着你这里的 GRPO 不是纯粹“自己探索自己打分”，而是一个明显带教师锚点的 group-relative policy optimization。

## 2. 为什么效果会好

## 2.1 训练目标和推理目标高度一致

这是第一层原因，而且非常关键。

训练和推理共用了几乎同一套任务接口：

- 都是相同的 prompt builder 思路；
- 都使用 `<PATH>` / `</PATH>` 包围路径；
- 都兼容 chat template 下 `<PATH>` 贴在 assistant generation 开头的写法；
- 都是在路径生成任务上直接优化，而不是绕一层答案生成再间接学路径。

这种“任务形式一致性”很重要，因为很多微调失败，本质上都失败在 train-test mismatch 上。  
而你这套实现中，SFT 数据构造、GRPO rollout prompt、主推理脚本的路径生成形式是高度对齐的。

## 2.2 SFT 先把模型拉到“会生成路径”的局部最优附近

SFT 的作用不是单纯提高准确率，更重要的是降低后续策略优化的难度。

如果没有 SFT，GRPO 一开始面临的问题通常有四个：

- 模型可能不会稳定输出合法路径格式；
- 模型可能根本不善于在 prompt 末尾接 `<PATH>` 风格的结构化内容；
- rollout 大量是低质量样本，reward 噪声大；
- 相对优势信号会很弱，因为整组样本都很差。

而 SFT 先做完之后，模型至少已经具备：

- 生成路径语法的能力；
- 生成常见 KG 路径模板的能力；
- 将问题映射到路径模式的基本能力。

这让 GRPO 阶段不必从“不会生成路径”开始学，而是直接在“已经会生成某些可用路径”的基础上做偏好调整。

这本质上是一个经典现象：

> RL 在好初始化上效果会非常好，在差初始化上通常不稳定。

你的实现刚好给了 RL 一个非常好的初始化。

## 2.3 GRPO 并不是纯自由探索，而是“图约束 + 教师锚点”的受控优化

这是第二个特别重要的原因。

从实现上看，GRPO 阶段有三层强约束：

1. 图约束  
   rollout 阶段通过 trie 约束 token 生成，模型不能随便编路径，见 `finetune/grpo_trainer.py:260-264`。

2. GT 注入  
   每个 group 在生成前会先注入最多 1-2 条 ground-truth path，见 `finetune/grpo_trainer.py:303-320`。

3. 去重与屏蔽  
   已生成或已注入的路径会被屏蔽掉，再去补新的路径，见 `finetune/grpo_trainer.py:334-350`。

这三层机制叠加后，策略优化问题被显著简化了：

- action space 被图结构硬性缩小；
- group 中永远更容易出现高 reward anchor；
- 组内比较不是一堆随机垃圾样本，而是“好样本 + 一些可比但更差的样本”。

这会让 advantage 学习信号更稳定，也更有方向性。

换句话说，这里的 GRPO 之所以有效，很大程度上不是因为“探索很强”，而是因为“探索被控制得很好”。

## 2.4 GT 注入让 GRPO 更像“带正例锚点的偏好学习”

我认为这是你这套实现里最有意思、也最能解释效果的点。

`generate_group_paths()` 在真正采样前，先把最多 1-2 条 GT path 直接放进 group，见 `finetune/grpo_trainer.py:303-320`。这会带来三个直接好处：

### 第一，reward 尺度被锚定

group 中只要存在 GT path，高分样本就不会缺席。  
这让 reward 的上界和组内相对排序更稳定。

### 第二，advantage 不容易塌缩

如果整组 rollout 都很差，那标准化 advantage 的信号会很弱，甚至全都差不多。  
而 GT 注入让“明显好的样本”几乎总在组里，advantage 学习会更稳定。

### 第三，策略更新更像“向高分路径靠拢”

因为 group 里总有高质量教师样本，优化方向会明显偏向：

- 更像 GT 的结构；
- 更像 GT 的关系序列；
- 更像 GT 的终点实体。

这使得你的 GRPO 和纯 on-policy RL 不太一样。它更像：

> SFT 之后，再做一轮带在线负样本和相对奖励的偏好蒸馏。

这类训练通常会比纯 RL 稳定很多。

## 2.5 reward 是密集的，而且和任务高度对齐

当前 reward 设计不是单点奖励，而是一个多项组合：

- 终点实体命中答案：`+5.0`，见 `finetune/reward.py:299`
- 路径结构匹配：exact / prefix / relation LCS，见 `finetune/reward.py:202-224`
- 绕路惩罚：额外 hop 会扣分，见 `finetune/reward.py:323-324`
- 回环惩罚：重复访问实体会扣分，见 `finetune/reward.py:302-311`
- 语义辅助项：用 embedding 相似度做弱语义奖励，见 `finetune/reward.py:239-265`

这套 reward 好的地方在于，它不是只看“终点对不对”，也不是只看“路径像不像 GT”，而是把两个维度都考虑了：

- 答案正确性；
- 路径质量。

这非常适合路径生成任务，因为真实好路径通常同时满足：

- 终点正确；
- 结构不乱；
- 不绕圈；
- 关系序列和真值大体一致；
- 即使不完全一致，语义上也接近。

也就是说，你这里的 reward 并不是稀疏 reward，而是一个比较强的 shaping reward。  
这会显著提高 RL 阶段的可学性。

## 2.6 组内标准化 advantage 很适合“同题多路径”任务

`calculate_group_rewards()` 会在每个问题对应的一组路径内部做标准化 advantage，见 `finetune/reward.py:358-370`。

这在当前任务里很合理，因为你的优化目标天然是相对的：

- 同一个问题下，哪条路径更好？
- 哪条路径更接近真值？
- 哪条路径更短、更准、更不绕圈？

这种任务其实比绝对打分更适合 group-relative learning。  
因为对于不同问题，reward 的绝对数值可能差异较大，但在同一问题内进行“相对排序”通常更稳定。

所以 GRPO 在这里不是随便套了一个 RL 算法，而是和任务形式是匹配的。

## 2.7 去重和重采样提高了组内信息量

GRPO 的效果很大程度上取决于 rollout 组的质量。  
如果同一组里全是重复路径，优势学习几乎没什么信息量。

你这里在 `generate_group_paths()` 里专门做了：

- 路径 canonicalize；
- 已见路径去重；
- 已生成路径 token 屏蔽；
- 候选 trie 过滤；
- 多轮补生成；
- 连续两轮无新增时提前停止。

对应代码在 `finetune/grpo_trainer.py:292-398`。

这带来的直接收益是：

- 一组样本里更可能出现多样化候选；
- reward 排序更有意义；
- advantage 不会被一堆重复路径冲淡。

对 group-based RL 而言，这种“信息增益式采样”是非常有帮助的。

## 2.8 LoRA + 小学习率 + 低轮数让更新更保守

SFT 和 GRPO 两阶段都用了相同风格的 LoRA 配置：

- `r=64`
- `alpha=128`
- `dropout=0.05`
- 只打在 attention 和 MLP 的关键投影层，见 `finetune/config.py:93-102` 和 `finetune/config.py:214-223`

同时，两阶段学习率都很低：`2e-6`，epoch 也不多：`2`，见 `finetune/config.py:44-47` 与 `finetune/config.py:167-170`。

这会带来一种很典型的效果：

- 模型不会剧烈偏离基座能力；
- SFT 先做格式和分布对齐；
- GRPO 再在这个基础上做细粒度偏好修正。

这种保守更新策略特别适合结构化生成任务，因为它通常不需要彻底重塑模型，只需要把输出分布往“更像好路径”的方向推一点。

## 2.9 你的实现里的 GRPO 实际上比“标准 GRPO”更温和

这也是我认为效果会不错的一个隐藏原因。

在 `train_step()` 里：

- `old_log_probs` 是在当前模型上 `no_grad` 算出来的，见 `finetune/grpo_trainer.py:452-454`
- `new_log_probs` 紧接着又在同一个模型参数上重新前向算一次，见 `finetune/grpo_trainer.py:455-460`

所以这里的 KL 项并不是严格意义上的“旧策略 vs 新策略”KL。  
在没有 optimizer step 的情况下，它通常会非常小，更多只会受到 dropout 噪声影响。

因此这个 loss 实际上更接近：

```text
loss ≈ -(advantage * logprob).mean()
```

也就是一种比较温和的 reward-weighted likelihood / REINFORCE 风格更新，而不是激进的 PPO 式约束优化。

这种实现从算法纯度上说不算标准，但从工程稳定性上说，反而可能是好事：

- 更新更平滑；
- 不容易出现策略崩塌；
- 更像是在 SFT 基础上做偏好微调。

这也解释了为什么你观察到效果不错而不是训练发散。

## 3. 一个更本质的理解

如果把这套两阶段流程抽象一下，它其实更像下面这个结构：

### 阶段一：SFT

让模型学会：

- 任务格式；
- 路径语言；
- 真值路径分布；
- prompt 到 path 的基本映射。

### 阶段二：GRPO

让模型进一步学会：

- 在合法路径空间里偏好更优路径；
- 更看重正确终点；
- 少绕路、少回环；
- 在多条可行路径里更偏向真值风格。

所以它有效，并不是因为 GRPO 替代了 SFT，而是因为：

> SFT 负责“学会做这件事”，GRPO 负责“学会把这件事做得更像你想要的样子”。

这正是两阶段训练最理想的分工。

## 4. 需要注意的实现细节和边界

虽然整体上我能理解为什么效果会好，但也有几个实现细节值得注意。

## 4.1 GT 注入会显著稳定训练，但也让训练更像 teacher-anchored RL

这不是坏事，反而很可能正是效果好的原因之一。  
但它意味着：你的 GRPO 训练并不是纯探索式 RL，而是带教师样本锚点的相对优化。

## 4.2 当前 KL 项不是真正的 old-policy KL

如前所述，这会让算法更像 reward-weighted likelihood。  
这可能提升稳定性，但如果后面你想把它写成严格 GRPO/PPO 论文实现，需要额外说明。

## 4.3 `gradient_accumulation_steps` 在 GRPO trainer 中没有真正使用

`GRPOConfig` 里有 `gradient_accumulation_steps`，见 `finetune/config.py:47`，但 `GRPOTrainer.train()` 和 `train_step()` 里没有对应的累计逻辑。  
这不会解释“为什么效果好”，但会影响你对 batch size 和更新频率的理解。

## 4.4 语义奖励依赖外部 embedding API

`compute_semantic_reward()` 通过外部 embedding API 获取向量，见 `finetune/reward.py:28-90`。  
这意味着：

- reward 稳定性受外部服务影响；
- embedding 模型版本会影响结果；
- 复现实验时需要把 API 和模型版本固定。

## 4.5 训练和主推理脚本的 path length 默认值不同

`finetune/config.py` 里 SFT 和 GRPO 默认 `index_path_length=3`，见 `finetune/config.py:53` 与 `finetune/config.py:175`。  
而你前面打开的主推理脚本 `scripts/graph_constrained_decoding.sh` 里是 `INDEX_LEN=2`。

这不一定有问题，但它意味着：

- 训练时模型见到的路径空间可能比某些推理设置更长；
- 如果推理时再切回更短路径空间，模型能力未必完全等价。

如果后续做更严格 ablation，最好把这件事单独控制住。

## 5. 结论

在当前实现下，“先 SFT，再 GRPO”效果不错，我认为主要不是因为某一个神奇技巧，而是因为你把几个通常很容易不稳定的环节都处理成了稳定版本：

1. SFT 先把任务分布和输出格式学稳了。
2. 训练和推理的 prompt / special token / 输出形式高度一致。
3. GRPO rollout 在图约束空间中进行，探索空间被极大缩小。
4. GT 注入让每个 group 都有高质量锚点。
5. reward 不是稀疏奖励，而是答案、结构、长度、回环、语义多项联合。
6. 去重和补生成让组内比较更有信息量。
7. 低学习率 LoRA 让更新更保守。
8. 当前实现里的“GRPO”实际上比标准 GRPO 更温和，因此更稳。

如果让我用一句话概括：

> 你的两阶段方案之所以有效，是因为它把一个原本很难做稳的 RL 路径生成问题，转化成了“先监督学会路径语言，再在图约束和教师锚点下做温和的偏好优化”。

这类训练范式在结构化生成任务里通常很容易出效果。
