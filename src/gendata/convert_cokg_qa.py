#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COKG_QA数据集转换脚本
将COKG_QA文件夹中的知识图谱和问答对数据转换为kg_qa_dataset.json的格式
"""

import json
import re
import os
from typing import List, Dict, Tuple, Set
from collections import deque


def load_full_graph(kg_file: str) -> Tuple[List[List[str]], Dict[str, List[int]]]:
    """
    加载完整知识图谱并构建索引

    Args:
        kg_file: KG文件路径 (JSONL格式)

    Returns:
        (graph, entity_index)
        - graph: 完整的graph列表，格式为 [[头, 关系, 尾], ...]
        - entity_index: 实体到三元组索引的映射，格式为 {实体: [三元组索引列表]}
    """
    print(f"正在加载知识图谱: {kg_file}")
    graph = []
    entity_index = {}  # 实体 -> 包含该实体的三元组索引列表

    with open(kg_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i + 1) % 100000 == 0:
                print(f"  已加载 {i + 1} 个三元组...")

            # 解析JSON: [头, 关系, 尾, [类型]]
            triple_with_type = json.loads(line.strip())

            # 只保留前3个元素: [头, 关系, 尾]
            triple = triple_with_type[:3]
            graph.append(triple)

            # 构建索引：头实体和尾实体都指向这个三元组
            head, relation, tail = triple
            if head not in entity_index:
                entity_index[head] = []
            entity_index[head].append(i)

            if tail not in entity_index:
                entity_index[tail] = []
            entity_index[tail].append(i)

    print(f"知识图谱加载完成，共 {len(graph)} 个三元组")
    print(f"实体索引构建完成，共 {len(entity_index)} 个实体")
    return graph, entity_index


def find_shortest_paths_bfs(start_entities: List[str], target_entities: List[str],
                            graph: List[List[str]], entity_index: Dict[str, List[int]],
                            max_depth: int = 5) -> Tuple[Set[int], Dict[str, int]]:
    """
    使用BFS查找路径：
    - 若最短路径 <= 2跳，则收集所有1跳和2跳路径
    - 若最短路径 > 2跳，则只保留一条最短路径

    Args:
        start_entities: 起始实体列表（问题实体）
        target_entities: 目标实体列表（答案实体）
        graph: 完整图谱
        entity_index: 实体索引
        max_depth: 多跳路径的最大搜索深度（默认5跳）

    Returns:
        (路径上的三元组索引集合, 每个答案实体的最短路径长度字典)
    """
    path_triple_indices = set()
    target_set = set(target_entities)
    answer_path_lengths = {}

    for start_entity in start_entities:
        if start_entity not in entity_index:
            continue

        # --- 阶段1：枚举所有1跳和2跳路径 ---
        short_path_targets = {}  # {目标实体: [[路径三元组索引列表], ...]}

        # 1跳路径
        for triple_idx in entity_index.get(start_entity, []):
            head, relation, tail = graph[triple_idx]
            next_entity = tail if head == start_entity else head
            if next_entity in target_set:
                short_path_targets.setdefault(next_entity, []).append([triple_idx])

        # 2跳路径
        for triple_idx1 in entity_index.get(start_entity, []):
            head1, relation1, tail1 = graph[triple_idx1]
            mid_entity = tail1 if head1 == start_entity else head1
            if mid_entity == start_entity:
                continue
            for triple_idx2 in entity_index.get(mid_entity, []):
                if triple_idx2 == triple_idx1:
                    continue
                head2, relation2, tail2 = graph[triple_idx2]
                next_entity = tail2 if head2 == mid_entity else head2
                if next_entity in target_set and next_entity != start_entity:
                    short_path_targets.setdefault(next_entity, []).append([triple_idx1, triple_idx2])

        # 将所有短路径加入结果
        for target_entity, paths in short_path_targets.items():
            depth = min(len(p) for p in paths)
            if target_entity not in answer_path_lengths or depth < answer_path_lengths[target_entity]:
                answer_path_lengths[target_entity] = depth
            for path in paths:
                path_triple_indices.update(path)

        # --- 阶段2：对未在2跳内找到的目标，用BFS找一条最短多跳路径 ---
        remaining_targets = target_set - set(short_path_targets.keys())
        if not remaining_targets:
            continue

        queue = deque([(start_entity, [])])
        visited = {start_entity}
        target_paths = {}  # {目标实体: (深度, 路径)}

        while queue:
            current_entity, path_indices = queue.popleft()

            if len(path_indices) >= max_depth:
                continue

            if current_entity in remaining_targets:
                if current_entity not in target_paths:
                    target_paths[current_entity] = (len(path_indices), path_indices[:])
                continue

            if current_entity in entity_index:
                for triple_idx in entity_index[current_entity]:
                    head, relation, tail = graph[triple_idx]
                    next_entity = tail if head == current_entity else head
                    if next_entity not in visited:
                        visited.add(next_entity)
                        queue.append((next_entity, path_indices + [triple_idx]))

        for target_entity, (depth, path) in target_paths.items():
            if target_entity not in answer_path_lengths or depth < answer_path_lengths[target_entity]:
                answer_path_lengths[target_entity] = depth
            path_triple_indices.update(path)

    return path_triple_indices, answer_path_lengths


def extract_k_hop_neighbors(entity: str, k: int, graph: List[List[str]],
                            entity_index: Dict[str, List[int]], max_neighbors: int = None) -> set:
    """
    提取实体的K跳邻居三元组索引（支持限制邻居数）

    Args:
        entity: 起始实体
        k: 跳数
        graph: 完整图谱
        entity_index: 实体索引
        max_neighbors: 每个实体的最大邻居数限制，None表示不限制

    Returns:
        三元组索引的集合
    """
    if entity not in entity_index:
        return set()

    triple_indices = set()
    current_entities = {entity}

    for hop in range(k):
        next_entities = set()
        for ent in current_entities:
            if ent in entity_index:
                # 获取包含该实体的所有三元组索引
                ent_triple_indices = entity_index[ent]

                # 如果设置了最大邻居数限制，则只取前max_neighbors个
                if max_neighbors is not None and len(ent_triple_indices) > max_neighbors:
                    ent_triple_indices = ent_triple_indices[:max_neighbors]

                for idx in ent_triple_indices:
                    triple_indices.add(idx)
                    # 获取三元组中的其他实体作为下一跳的起点
                    head, relation, tail = graph[idx]
                    next_entities.add(head)
                    next_entities.add(tail)
        current_entities = next_entities

    return triple_indices


def extract_subgraph(q_entities: List[str], a_entities: List[str],
                     graph: List[List[str]], entity_index: Dict[str, List[int]],
                     q_hops: int = 1, a_hops: int = 1, max_neighbors: int = 100) -> Tuple[List[List[str]], int, int, int]:
    """
    提取子图：先添加最短路径，再基于问题实体和答案实体扩展（混合策略）

    Args:
        q_entities: 问题实体列表
        a_entities: 答案实体列表
        graph: 完整图谱
        entity_index: 实体索引
        q_hops: 问题实体的跳数（默认1跳）
        a_hops: 答案实体的跳数（默认1跳）
        max_neighbors: 每个实体的最大邻居数限制（默认100）

    Returns:
        (子图三元组列表, 最短路径三元组数量, 找到路径的答案数量, 总答案数量)
    """
    triple_indices = set()

    # 1. 首先查找并添加从问题实体到答案实体的最短路径
    path_indices, answer_path_lengths = find_shortest_paths_bfs(q_entities, a_entities, graph, entity_index, max_depth=5)
    triple_indices.update(path_indices)
    path_count = len(path_indices)

    # 统计答案覆盖情况
    answers_found = len(answer_path_lengths)  # 找到路径的答案数量
    total_answers = len(a_entities)  # 总答案数量

    # 2. 提取问题实体的K跳邻居（限制邻居数）
    for q_entity in q_entities:
        indices = extract_k_hop_neighbors(q_entity, q_hops, graph, entity_index, max_neighbors)
        triple_indices.update(indices)

    # 3. 提取答案实体的K跳邻居（限制邻居数）
    for a_entity in a_entities:
        indices = extract_k_hop_neighbors(a_entity, a_hops, graph, entity_index, max_neighbors)
        triple_indices.update(indices)

    # 根据索引提取三元组
    subgraph = [graph[idx] for idx in sorted(triple_indices)]
    return subgraph, path_count, answers_found, total_answers


def parse_qa_line(line: str) -> Dict:
    """
    解析单行QA数据

    Args:
        line: QA文件的一行，格式为: 问题\t答案\t类型

    Returns:
        包含question, answers, q_entities的字典
    """
    parts = line.strip().split('\t')

    if len(parts) < 2:
        return None

    question = parts[0]
    answer_str = parts[1]
    # parts[2]是类型信息，我们不需要

    # 提取问题实体（方括号内的内容）
    q_entities = re.findall(r'\[(.*?)\]', question)

    # 分割答案（用##分隔）
    answers = answer_str.split('##')

    return {
        'question': question,
        'answers': answers,
        'q_entities': q_entities
    }


def convert_dataset(qa_file: str, full_graph: List[List[str]], entity_index: Dict[str, List[int]],
                    split_name: str, max_samples: int = None, output_file: str = None):
    """
    转换单个数据集并直接写入文件（JSONL格式）

    Args:
        qa_file: QA文件路径
        full_graph: 完整的知识图谱
        entity_index: 实体索引
        split_name: 数据集名称 (train/test/valid)
        max_samples: 最大样本数，None表示全部
        output_file: 输出文件路径
    """
    print(f"\n正在转换数据集: {split_name}")
    if max_samples:
        print(f"  限制样本数: {max_samples}")
    print(f"  输出文件: {output_file}")

    count = 0
    total_subgraph_size = 0
    total_path_triples = 0
    samples_with_path = 0
    total_answers_found = 0  # 总共找到路径的答案数量
    total_answers_count = 0  # 总答案数量

    with open(qa_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            # 如果达到最大样本数，停止处理
            if max_samples and i >= max_samples:
                break

            # 解析QA行
            parsed = parse_qa_line(line)
            if parsed is None:
                continue

            # 提取子图（先添加最短路径，再扩展邻居：问题实体2跳 + 答案实体2跳 + 限制邻居数200）
            subgraph, path_count, answers_found, total_answers = extract_subgraph(
                q_entities=parsed['q_entities'],
                a_entities=parsed['answers'],
                graph=full_graph,
                entity_index=entity_index,
                q_hops=2,
                a_hops=2,
                max_neighbors=200
            )

            # 统计路径信息
            if path_count > 0:
                samples_with_path += 1
                total_path_triples += path_count

            # 统计答案覆盖情况
            total_answers_found += answers_found
            total_answers_count += total_answers

            # 生成数据项
            data_item = {
                'id': f'cokg_qa_{split_name}_{i + 1}',
                'question': parsed['question'],
                'answer': parsed['answers'],
                'q_entity': parsed['q_entities'],
                'a_entity': parsed['answers'],  # 答案实体与答案相同
                'graph': subgraph  # 使用子图而不是完整图谱
            }

            # 写入JSONL格式（每行一个JSON对象）
            f_out.write(json.dumps(data_item, ensure_ascii=False) + '\n')
            count += 1
            total_subgraph_size += len(subgraph)

            # 显示保存进度
            if count % 50 == 0:
                # 获取当前文件大小
                f_out.flush()  # 确保数据写入磁盘
                current_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                avg_subgraph_size = total_subgraph_size / count
                avg_path_triples = total_path_triples / samples_with_path if samples_with_path > 0 else 0
                path_coverage = (samples_with_path / count * 100) if count > 0 else 0
                answer_coverage = (total_answers_found / total_answers_count * 100) if total_answers_count > 0 else 0
                print(f"  已保存 {count} 条数据，当前文件大小: {current_size:.2f} MB")
                print(f"    平均子图大小: {avg_subgraph_size:.1f} 个三元组")
                print(f"    样本路径覆盖率: {path_coverage:.1f}% ({samples_with_path}/{count})")
                print(f"    答案实体覆盖率: {answer_coverage:.1f}% ({total_answers_found}/{total_answers_count})")
                print(f"    平均路径长度: {avg_path_triples:.1f} 个三元组")

    avg_subgraph_size = total_subgraph_size / count if count > 0 else 0
    avg_path_triples = total_path_triples / samples_with_path if samples_with_path > 0 else 0
    path_coverage = (samples_with_path / count * 100) if count > 0 else 0
    answer_coverage = (total_answers_found / total_answers_count * 100) if total_answers_count > 0 else 0

    print(f"数据集 {split_name} 转换完成:")
    print(f"  总样本数: {count}")
    print(f"  平均子图大小: {avg_subgraph_size:.1f} 个三元组")
    print(f"  样本路径覆盖率: {path_coverage:.1f}% ({samples_with_path}/{count})")
    print(f"  答案实体覆盖率: {answer_coverage:.1f}% ({total_answers_found}/{total_answers_count})")
    print(f"  平均路径长度: {avg_path_triples:.1f} 个三元组")

    return count, samples_with_path, path_coverage, answer_coverage


def main():
    """主函数"""
    print("=" * 60)
    print("COKG_QA数据集转换工具")
    print("=" * 60)

    # 定义文件路径
    kg_file = 'COKG_QA/KG/one_hop_KG.json'
    qa_files = {
        'train': 'COKG_QA/QA/three_hop/train.txt',
        'test': 'COKG_QA/QA/three_hop/test.txt',
        'valid': 'COKG_QA/QA/three_hop/valid.txt'
    }
    # 每个数据集的样本数限制
    max_samples = {
        'train': 500,
        'test': 200,
        'valid': 100
    }
    output_dir = 'COKG_QA/threehop'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 1. 加载完整知识图谱并构建索引
    full_graph, entity_index = load_full_graph(kg_file)

    # 2. 转换每个数据集
    results = {}
    for split_name, qa_file in qa_files.items():
        # 输出文件路径（JSONL格式）
        output_file = os.path.join(output_dir, f'cokg_qa_{split_name}.jsonl')

        # 转换数据集并直接写入文件
        count, samples_with_path, path_coverage, answer_coverage = convert_dataset(
            qa_file, full_graph, entity_index, split_name, max_samples[split_name], output_file
        )

        # 获取最终文件大小
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"[完成] 文件已保存完成，最终大小: {file_size:.2f} MB\n")

        results[split_name] = {
            'count': count,
            'size': file_size,
            'samples_with_path': samples_with_path,
            'path_coverage': path_coverage,
            'answer_coverage': answer_coverage
        }

    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)

    # 输出统计信息
    print("\n统计信息:")
    print(f"  知识图谱三元组数: {len(full_graph)}")
    for split_name, info in results.items():
        print(f"  {split_name}:")
        print(f"    样本数: {info['count']}")
        print(f"    文件大小: {info['size']:.2f} MB")
        print(f"    样本路径覆盖率: {info['path_coverage']:.1f}% ({info['samples_with_path']}/{info['count']})")
        print(f"    答案实体覆盖率: {info['answer_coverage']:.1f}%")


if __name__ == '__main__':
    main()
