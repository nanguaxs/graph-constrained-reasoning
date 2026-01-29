import json
import random
from collections import defaultdict

# 读取原始知识图谱
with open('kg_triples_clean.json', 'r', encoding='utf-8') as f:
    original_triples = json.load(f)

print(f"原始三元组数量: {len(original_triples)}")

# 提取所有实体和关系
entities = set()
relations = set()
for triple in original_triples:
    entities.add(triple[0])
    entities.add(triple[2])
    relations.add(triple[1])

entities = list(entities)
print(f"实体数量: {len(entities)}")
print(f"现有关系类型: {relations}")

# 构建实体索引，方便查找相关实体
entity_connections = defaultdict(set)
for triple in original_triples:
    entity_connections[triple[0]].add(triple[2])
    entity_connections[triple[2]].add(triple[0])

# 定义新的关系类型
new_relations = [
    "依赖于",      # 表示依赖关系
    "竞争于",      # 表示竞争关系
    "服务于",      # 表示服务关系
    "监管",        # 表示监管关系
    "协作于",      # 表示协作关系
    "影响",        # 表示影响关系
    "支持",        # 表示支持关系
    "促进",        # 表示促进关系
    "制约",        # 表示制约关系
    "关联于",      # 表示关联关系
    "派生于",      # 表示派生关系
    "融合于",      # 表示融合关系
    "转型为",      # 表示转型关系
    "供应给",      # 表示供应关系
    "需要",        # 表示需求关系
    "创新于",      # 表示创新关系
    "整合",        # 表示整合关系
    "优化",        # 表示优化关系
    "替代",        # 表示替代关系
    "补充",        # 表示补充关系
]

print(f"新增关系类型: {new_relations}")

# 生成新的三元组
new_triples = []
target_count = 3000

# 策略1: 为现有实体对添加新关系 (40%)
strategy1_count = int(target_count * 0.4)
for _ in range(strategy1_count):
    entity1 = random.choice(entities)
    entity2 = random.choice(entities)
    if entity1 != entity2:
        relation = random.choice(new_relations)
        new_triples.append([entity1, relation, entity2])

# 策略2: 基于已有连接的实体，添加间接关系 (30%)
strategy2_count = int(target_count * 0.3)
for _ in range(strategy2_count):
    entity1 = random.choice(entities)
    if entity1 in entity_connections and len(entity_connections[entity1]) > 0:
        # 找到与entity1相关的实体
        connected = list(entity_connections[entity1])
        entity2 = random.choice(connected)
        # 添加新的关系类型
        relation = random.choice(new_relations)
        new_triples.append([entity1, relation, entity2])
    else:
        entity2 = random.choice(entities)
        if entity1 != entity2:
            relation = random.choice(new_relations)
            new_triples.append([entity1, relation, entity2])

# 策略3: 创建行业间的关系 (30%)
strategy3_count = target_count - strategy1_count - strategy2_count
# 识别可能的行业实体（包含"业"字的）
industries = [e for e in entities if "业" in e or "服务" in e or "生产" in e]
if len(industries) < 2:
    industries = entities[:100]  # 如果没有足够的行业，使用前100个实体

for _ in range(strategy3_count):
    if len(industries) >= 2:
        entity1 = random.choice(industries)
        entity2 = random.choice(industries)
        if entity1 != entity2:
            relation = random.choice(new_relations)
            new_triples.append([entity1, relation, entity2])
    else:
        entity1 = random.choice(entities)
        entity2 = random.choice(entities)
        if entity1 != entity2:
            relation = random.choice(new_relations)
            new_triples.append([entity1, relation, entity2])

print(f"生成的新三元组数量: {len(new_triples)}")

# 去重
new_triples_set = set()
unique_new_triples = []
for triple in new_triples:
    triple_tuple = tuple(triple)
    if triple_tuple not in new_triples_set:
        new_triples_set.add(triple_tuple)
        unique_new_triples.append(triple)

print(f"去重后的新三元组数量: {len(unique_new_triples)}")

# 合并原始三元组和新三元组
expanded_triples = original_triples + unique_new_triples

print(f"扩充后的总三元组数量: {len(expanded_triples)}")

# 保存到新文件
output_file = 'kg_triples_expanded.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(expanded_triples, f, ensure_ascii=False, indent=2)

print(f"已保存到文件: {output_file}")

# 统计信息
print("\n=== 统计信息 ===")
print(f"原始三元组: {len(original_triples)}")
print(f"新增三元组: {len(unique_new_triples)}")
print(f"总计三元组: {len(expanded_triples)}")

# 统计新关系的使用情况
new_relation_count = defaultdict(int)
for triple in unique_new_triples:
    new_relation_count[triple[1]] += 1

print("\n新关系类型使用统计:")
for rel, count in sorted(new_relation_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {rel}: {count}")
