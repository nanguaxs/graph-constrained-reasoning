# GCR 项目中的两种 Trie 数据结构分析

## 1. 为什么要用 Trie(前缀树)而不是普通树?

### 问题场景
在图约束解码过程中,需要实时判断:
- **当前已生成的 token 序列** + **下一个候选 token** 是否能构成知识图谱中存在的路径

### 普通树 vs Trie 对比

**如果用普通树/列表存储:**
```python
# 假设有 100,000 条 KG 路径
kg_paths = [
    [token1, token2, token3, token4],
    [token1, token2, token5, token6],
    [token1, token7, token8, token9],
    ...
]

# 每次解码时需要检查: 已生成 [token1, token2] 后,能生成哪些 token?
def get_allowed_tokens(prefix):
    allowed = set()
    for path in kg_paths:  # 遍历所有路径 O(N)
        if path[:len(prefix)] == prefix:  # 前缀匹配 O(M)
            if len(path) > len(prefix):
                allowed.add(path[len(prefix)])
    return allowed

# 时间复杂度: O(N * M), N=路径数量, M=前缀长度
# 每生成一个 token 都要遍历所有路径!
```

**使用 Trie 前缀树:**
```python
# Trie 结构示例
trie = {
    token1: {
        token2: {
            token3: {token4: {}},
            token5: {token6: {}}
        },
        token7: {
            token8: {token9: {}}
        }
    }
}

# 查询允许的 token
def get_allowed_tokens(prefix):
    node = trie
    for token in prefix:  # O(M)
        node = node[token]
    return list(node.keys())  # O(K), K=分支数

# 时间复杂度: O(M + K), 不需要遍历所有路径!
```

### 性能差异
- **KG 中有 100万 条路径,每条长度 10**
- 普通方法: 每次查询 ~1000万次比较
- Trie 方法: 每次查询 ~10-100 次查找

**在 Beam Search 中每个 token 都要查询,差异会被放大数千倍!**

---

## 2. 两种 Trie 实现详解

## 2.1 原生 Python Trie

```python
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}  # 嵌套字典实现
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
```

### 核心方法实现

#### 1. 添加序列 (递归构建)
```python
@staticmethod
def _add_to_trie(sequence: List[int], trie_dict: Dict):
    if sequence:
        if sequence[0] not in trie_dict:
            trie_dict[sequence[0]] = {}  # 创建子节点
        Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])  # 递归添加
```

**示例:**
```python
# 添加路径 [1, 2, 3]
trie_dict = {}
_add_to_trie([1, 2, 3], trie_dict)
# 结果: {1: {2: {3: {}}}}

# 再添加 [1, 2, 4]
_add_to_trie([1, 2, 4], trie_dict)
# 结果: {1: {2: {3: {}, 4: {}}}}  # 共享前缀 [1, 2]
```

#### 2. 查询允许的 token (核心!)
```python
@staticmethod
def _get_from_trie(prefix_sequence: List[int], trie_dict: Dict, ...):
    if len(prefix_sequence) == 0:
        # 前缀匹配完成,返回所有子节点的 key
        return list(trie_dict.keys())
    elif prefix_sequence[0] in trie_dict:
        # 继续递归匹配下一个 token
        return Trie._get_from_trie(
            prefix_sequence[1:], 
            trie_dict[prefix_sequence[0]], ...
        )
    else:
        return []  # 前缀不存在
```

**示例:**
```python
trie_dict = {1: {2: {3: {}, 4: {}}}}

get([])      # 返回 [1]  - 路径开始只能是 1
get([1])     # 返回 [2]  - 生成了 [1] 后只能接 2
get([1, 2])  # 返回 [3, 4]  - 生成了 [1,2] 后可以接 3 或 4
get([1, 5])  # 返回 []  - 不存在这样的路径
```

### 优点
- 实现简单,易于调试
- 灵活,可以动态添加路径

### 缺点
- **内存占用大**: Python 字典对象开销大
- **速度较慢**: 多层嵌套字典查找有开销

---

## 2.2 MarisaTrie (优化版)

```python
class MarisaTrie(object):
    def __init__(self, sequences: List[List[int]] = [], ...):
        # 将整数 token 映射为字符
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + ...
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}
        
        # 使用 C 实现的 marisa_trie 库
        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) 
            for sequence in sequences
        )
```

### 关键设计

#### 1. Token 到字符的映射
```python
# 为什么要做这个转换?
sequences = [[100, 200, 300], [100, 200, 400]]

# marisa_trie 只支持字符串,不支持整数序列
# 所以需要转换:
# [100, 200, 300] -> chr(100) + chr(200) + chr(300) -> "某字符串"

# 项目中跳过了 55000-65000 之间的字符,因为这是 Unicode 代理区,会出问题
```

#### 2. 查询实现
```python
def get(self, prefix_sequence: List[int]):
    if len(prefix_sequence) == 0:
        # 缓存第一层分支,常见查询
        return self.zero_iter
    else:
        # 将整数前缀转为字符串
        key = "".join([self.int2char[i] for i in prefix_sequence])
        # 查询所有以 key 开头的字符串
        return list({
            self.char2int[e[len(key)]]  # 提取下一个字符
            for e in self.trie.keys(key)  # marisa_trie 的前缀查询
            if len(e) > len(key)
        })
```

**示例:**
```python
# 内部存储: ["abc", "abd", "xyz"]  (假设 a=1, b=2, c=3, d=4)

get([])      # 返回 [1, 24]  - 'a' 和 'x' 对应的整数
get([1])     # 返回 [2]  - 以 'a' 开头后只能是 'b'
get([1, 2])  # 返回 [3, 4]  - "ab" 后可以是 'c' 或 'd'
```

### 优点
- **内存效率高**: marisa-trie 使用 MARISA (Matching Algorithm with Recursively Implemented StorAge)
  - 压缩存储,共享公共前缀/后缀
  - 10万条路径可能只占用几 MB
- **查询速度快**: C++ 实现,高度优化
- **只读优化**: 一次构建,多次查询

### 缺点
- 不能动态添加 (只读)
- 需要额外的编码/解码开销
- 调试困难

---

## 3. 两种 Trie 的使用场景对比

| 特性 | Python Trie | MarisaTrie |
|------|-------------|------------|
| **构建速度** | 快 | 慢 (需要构建整个数据结构) |
| **内存占用** | 大 (10-100x) | 小 (压缩存储) |
| **查询速度** | 较慢 | 快 (C++ 实现) |
| **动态修改** | ✅ 支持 | ❌ 只读 |
| **适用规模** | 小规模 (<1万路径) | 大规模 (>10万路径) |

### 项目中的选择

看代码中主要使用 **MarisaTrie**:

```python
# src/qa_prompt_builder.py line 71
return MarisaTrie(tokenized_path_list, max_token_id=len(self.tokenizer) + 1)
```

**原因:**
1. **知识图谱路径数量巨大**: WebQSP/CWQ 数据集中,每个问题对应的 KG 子图可能有数万到数十万条路径
2. **推理阶段查询频繁**: Beam Search 中每个 step、每个 beam 都要查询
3. **不需要动态修改**: 路径在推理前已经固定

---

## 4. 在 Graph Constrained Decoding 中的应用

### 完整流程

```python
# 1. 构建 Trie (预处理)
def get_graph_index(self, question_dict):
    # 从 KG 中提取所有可能的路径 (DFS)
    paths_list = utils.dfs(g, start_nodes, max_length=2)
    
    # 将路径转为 token 序列
    paths_list_str = [utils.path_to_string(p) for p in paths_list]
    tokenized_paths = self.tokenizer(paths_list_str).input_ids
    
    # 构建 MarisaTrie
    return MarisaTrie(tokenized_paths, max_token_id=len(self.tokenizer) + 1)

# 2. 解码时使用 Trie 约束
def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
    # 提取已生成的 <PATH> 和 </PATH> 之间的 token
    if constrained_flag:
        allow_tokens = self.trie.get(sent.tolist()[L_input:])
        # trie.get() 返回所有允许的下一个 token
        return allow_tokens
    return self.all_tokens  # 不在 <PATH> 内,不约束
```

### 实际例子

```
Question: "Where was Barack Obama born?"
Topic Entity: "Barack Obama"

KG 中的路径 (简化):
1. Barack Obama -> birthplace -> Honolulu
2. Barack Obama -> birthplace -> Hawaii
3. Barack Obama -> profession -> Politician

Tokenizer 后:
1. [2045, 8084, 4287, 98623, 6254, 24671]
2. [2045, 8084, 4287, 98623, 6254, 12345]  
3. [2045, 8084, 4287, 67890, 12121, 45678]

MarisaTrie 存储后:
{
  2045: {
    8084: {
      4287: {
        98623: {6254: {24671: {}, 12345: {}}},
        67890: {12121: {45678: {}}}
      }
    }
  }
}

LLM 生成过程:
已生成: <PATH> Barack Obama -> birthplace
Token序列: [2045, 8084, 4287, 98623, 6254]

查询 trie.get([2045, 8084, 4287, 98623, 6254])
返回: [24671, 12345]  (只能是 Honolulu 或 Hawaii)

如果 LLM 想生成 "Politician" (token 45678) -> 被阻止!
因为 45678 不在允许列表中
```

---

## 5. 总结

### 为什么用 Trie?
**核心原因: 前缀共享 + 快速前缀查询**

知识图谱路径有大量公共前缀:
```
Barack Obama -> birthplace -> Honolulu
Barack Obama -> birthplace -> Hawaii
Barack Obama -> birthplace -> Chicago  (错误答案)
Barack Obama -> birthdate -> 1961
...
(共享 "Barack Obama" 前缀)
```

### 为什么两种实现?
1. **Python Trie**: 开发/调试阶段,小规模测试
2. **MarisaTrie**: 生产环境,大规模高效推理

### 关键价值
通过 Trie 实现了 **"硬约束"**:
- LLM 不可能生成 KG 中不存在的路径
- 实现 "零推理幻觉" 的关键技术
- 比后处理过滤效率高 1000 倍

这就是为什么这个项目叫 "Graph-**constrained** Reasoning"!
