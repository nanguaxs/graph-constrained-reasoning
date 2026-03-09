"""奖励函数：终点命中 + 语义相似度 + 绕路惩罚"""
import torch
import re
import requests

class PathRewardCalculator:
    def __init__(self, embedding_api_url="http://localhost:8000/embed"):
        """
        Args:
            embedding_api_url: 嵌入模型 API 地址
        """
        self.embedding_api_url = embedding_api_url

    def get_embeddings(self, texts):
        """调用服务器 API 获取嵌入向量"""
        response = requests.post(
            self.embedding_api_url,
            json={"texts": texts},
            timeout=30
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        return torch.tensor(embeddings, dtype=torch.float32)

    def extract_final_entity(self, path_text):
        """从路径文本中提取终点实体"""
        print(f"\n[DEBUG] 原始路径文本: {repr(path_text)}")

        # 匹配最后一个 -> 后的实体
        matches = re.findall(r'->\s*([^-]+?)(?:\s*->|</PATH>|$)', path_text)
        print(f"[DEBUG] 正则匹配结果: {matches}")

        final_entity = matches[-1].strip() if matches else None
        print(f"[DEBUG] 提取的终点实体: {repr(final_entity)}\n")

        return final_entity

    def calculate_reward(self, generated_path, a_entity, expected_hops=2):
        """
        计算单条路径的奖励
        Args:
            generated_path: 生成的路径文本
            a_entity: 目标答案实体（可能是列表）
            expected_hops: 预期跳数
        """
        print(f"\n{'='*60}")
        print(f"[奖励计算] 生成路径: {generated_path}")
        print(f"[奖励计算] 目标答案: {a_entity}")

        final_entity = self.extract_final_entity(generated_path)
        if not final_entity:
            print(f"[奖励计算] 未提取到终点实体，返回 -1.0")
            return -1.0

        # 确保 a_entity 是列表
        if isinstance(a_entity, str):
            a_entity = [a_entity]

        # 1. 终点命中奖励
        if final_entity in a_entity:
            print(f"[奖励计算] ✓ 终点命中！奖励 = 10.0")
            print(f"{'='*60}\n")
            return 10.0

        # 2. 语义相似度奖励
        final_emb = self.get_embeddings([final_entity])
        answer_embs = self.get_embeddings(a_entity)
        similarities = torch.cosine_similarity(final_emb, answer_embs, dim=1)
        max_sim = similarities.max().item()
        reward = max_sim * 3.0
        print(f"[奖励计算] 语义相似度: {max_sim:.4f}, 相似度奖励: {reward:.4f}")

        # 3. 绕路惩罚
        hops = generated_path.count('->')
        if hops > expected_hops:
            penalty = (hops - expected_hops) * 0.5
            reward -= penalty
            print(f"[奖励计算] 跳数: {hops}, 预期: {expected_hops}, 绕路惩罚: -{penalty:.4f}")

        print(f"[奖励计算] 最终奖励: {reward:.4f}")
        print(f"{'='*60}\n")
        return reward

    def calculate_group_rewards(self, generated_paths, a_entity, expected_hops=2):
        """计算一组路径的奖励并标准化"""
        rewards = [self.calculate_reward(p, a_entity, expected_hops) for p in generated_paths]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 标准化为优势值
        if len(rewards) > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        return rewards, advantages
