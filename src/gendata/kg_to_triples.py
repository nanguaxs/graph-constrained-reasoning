#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱转换为三元组列表 - 过滤缺失节点版本
"""

import pandas as pd
import json
from typing import List, Tuple, Dict


def load_kg_from_csv(nodes_file: str, relations_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载知识图谱的节点和关系CSV文件"""
    nodes_df = pd.read_csv(nodes_file)
    relations_df = pd.read_csv(relations_file)
    return nodes_df, relations_df


def build_id_to_name_mapping(nodes_df: pd.DataFrame) -> Dict[float, str]:
    """构建节点ID到名称的映射，跳过名称为NaN的节点"""
    id_to_name = {}
    
    for _, row in nodes_df.iterrows():
        node_id = row['_id']
        node_name = row['name']
        
        # 跳过NaN节点
        if pd.isna(node_name):
            continue
        
        id_to_name[node_id] = node_name
    
    return id_to_name


def convert_to_triples(relations_df: pd.DataFrame, id_to_name: Dict[float, str]) -> List[List[str]]:
    """将关系转换为三元组列表，跳过实体缺失的三元组"""
    triples = []
    
    for _, row in relations_df.iterrows():
        start_id = row[':START_ID']
        end_id = row[':END_ID']
        rel_type = row[':TYPE']
        
        # 获取实体名称
        head_entity = id_to_name.get(start_id)
        tail_entity = id_to_name.get(end_id)
        
        # 跳过实体缺失的三元组
        if head_entity is None or tail_entity is None:
            continue
        
        # 处理关系类型
        if rel_type.lower() == 'weight' and 'weight' in row and pd.notna(row['weight']):
            relation = str(row['weight'])
        else:
            relation = rel_type
        
        triple = [head_entity, relation, tail_entity]
        triples.append(triple)
    
    return triples


def save_triples(triples: List[List[str]], output_file: str, format: str = 'json'):
    """保存三元组到文件"""
    if format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)
    elif format == 'txt':
        with open(output_file, 'w', encoding='utf-8') as f:
            for head, rel, tail in triples:
                f.write(f"{head}\t{rel}\t{tail}\n")


def main():
    """主函数"""
    # 文件路径
    nodes_file = "nodes.csv"
    relations_file = "relationships.csv"
    
    # 输出文件
    output_json = "kg_triples_clean.json"
    output_txt = "kg_triples_clean.txt"
    
    # 加载数据
    nodes_df, relations_df = load_kg_from_csv(nodes_file, relations_file)
    
    # 构建映射（跳过NaN节点）
    id_to_name = build_id_to_name_mapping(nodes_df)
    
    # 转换为三元组（跳过缺失实体）
    triples = convert_to_triples(relations_df, id_to_name)
    
    # 保存结果
    save_triples(triples, output_json, format='json')
    save_triples(triples, output_txt, format='txt')
    
    print(f"生成三元组数: {len(triples)}")
    print(f"保存到: {output_json}, {output_txt}")
    
    return triples


if __name__ == "__main__":
    triples = main()
