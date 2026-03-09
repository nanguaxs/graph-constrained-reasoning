"""奖励函数：终点命中 + 问答组合语义相似度 + 绕路惩罚"""
import re

import requests
import torch


class PathRewardCalculator:
    def __init__(
        self,
        embedding_api_url="http://localhost:8000/v1/embeddings",
        embedding_model_name="text-embedding-3-small",
        embedding_api_key="",
    ):
        """
        Args:
            embedding_api_url: 嵌入模型 API 地址
        """
        self.embedding_api_url = embedding_api_url
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.embedding_cache = {}

    def get_embeddings(self, texts):
        """调用服务端 API 获取嵌入向量，并缓存重复文本。"""
        normalized_texts = [str(text).strip() for text in texts]
        uncached_texts = [text for text in dict.fromkeys(normalized_texts) if text not in self.embedding_cache]

        if uncached_texts:
            headers = {"Content-Type": "application/json"}
            if self.embedding_api_key:
                headers["Authorization"] = f"Bearer {self.embedding_api_key}"

            response = requests.post(
                self.embedding_api_url,
                json={"input": uncached_texts, "model": self.embedding_model_name},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if "data" in payload:
                embeddings = [item["embedding"] for item in payload["data"]]
            else:
                embeddings = payload["embeddings"]

            for text, embedding in zip(uncached_texts, embeddings):
                self.embedding_cache[text] = torch.tensor(embedding, dtype=torch.float32)

        return torch.stack([self.embedding_cache[text] for text in normalized_texts])

    def parse_path_segments(self, path_text):
        """将路径文本解析为 [实体, 关系, 实体, ...] 的分段列表。"""
        print(f"\n[DEBUG] 原始路径文本: {repr(path_text)}")

        if not path_text:
            print("[DEBUG] 路径文本为空")
            return []

        normalized_text = str(path_text).strip()

        if "<PATH>" in normalized_text:
            normalized_text = normalized_text.rsplit("<PATH>", 1)[-1]

        end_markers = (
            "</PATH>",
            "\nAnswer:",
            "\r\nAnswer:",
            "\nthe answer is:",
            "\r\nthe answer is:",
            "\n答案：",
            "\r\n答案：",
            "\n答案:",
            "\r\n答案:",
        )
        for marker in end_markers:
            if marker in normalized_text:
                normalized_text = normalized_text.split(marker, 1)[0]
                break

        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        path_segments = [segment.strip() for segment in normalized_text.split("->") if segment.strip()]

        print(f"[DEBUG] 规范化路径文本: {repr(normalized_text)}")
        print(f"[DEBUG] 路径分段结果: {path_segments}")
        return path_segments

    def extract_final_entity(self, path_text):
        """从路径文本中提取终点实体。"""
        path_segments = self.parse_path_segments(path_text)
        if not path_segments:
            print("[DEBUG] 未解析出任何路径分段，终点实体为 None\n")
            return None

        if len(path_segments) % 2 == 0:
            print("[DEBUG] 路径分段数为偶数，说明路径停在关系上，终点实体为 None\n")
            return None

        final_entity = path_segments[-1]
        print(f"[DEBUG] 提取的终点实体: {repr(final_entity)}\n")
        return final_entity

    def count_hops(self, path_text):
        """根据解析后的路径段数统计 hop 数。"""
        path_segments = self.parse_path_segments(path_text)
        if len(path_segments) < 3:
            return 0
        return (len(path_segments) - 1) // 2

    def build_question_answer_texts(self, question, a_entity):
        """构造“问题 + 答案”语义参考文本列表。"""
        if isinstance(a_entity, str):
            a_entity = [a_entity]

        question_text = str(question).strip()
        answer_texts = [str(answer).strip() for answer in a_entity if str(answer).strip()]
        return [f"问题：{question_text} 答案：{answer_text}" for answer_text in answer_texts]

    def calculate_reward(self, generated_path, question, a_entity, expected_hops=2):
        """计算单条路径的奖励。"""
        print(f"\n{'=' * 60}")
        print(f"[奖励计算] 生成路径: {generated_path}")
        print(f"[奖励计算] 问题: {question}")
        print(f"[奖励计算] 目标答案: {a_entity}")

        path_segments = self.parse_path_segments(generated_path)
        if not path_segments:
            print("[奖励计算] 未解析出有效路径，返回 -1.0")
            return -1.0

        if len(path_segments) % 2 == 0:
            print("[奖励计算] 路径结构不完整，最后停在关系上，返回 -1.0")
            return -1.0

        final_entity = path_segments[-1]
        print(f"[DEBUG] 提取的终点实体: {repr(final_entity)}\n")

        if isinstance(a_entity, str):
            a_entity = [a_entity]

        if final_entity in a_entity:
            print("[奖励计算] ✓ 终点命中！奖励 = 10.0")
            print(f"{'=' * 60}\n")
            return 10.0

        path_text_for_similarity = " -> ".join(path_segments)
        question_answer_texts = self.build_question_answer_texts(question, a_entity)
        path_emb = self.get_embeddings([path_text_for_similarity])
        question_answer_embs = self.get_embeddings(question_answer_texts)
        similarities = torch.cosine_similarity(path_emb, question_answer_embs, dim=1)
        best_index = similarities.argmax().item()
        max_sim = similarities[best_index].item()
        reward = max_sim * 3.0

        print(f"[奖励计算] 语义参考文本: {question_answer_texts[best_index]}")
        print(f"[奖励计算] 语义相似度: {max_sim:.4f}, 相似度奖励: {reward:.4f}")

        hops = (len(path_segments) - 1) // 2
        if hops > expected_hops:
            penalty = (hops - expected_hops) * 0.5
            reward -= penalty
            print(f"[奖励计算] 跳数: {hops}, 预期: {expected_hops}, 绕路惩罚: -{penalty:.4f}")

        print(f"[奖励计算] 最终奖励: {reward:.4f}")
        print(f"{'=' * 60}\n")
        return reward

    def calculate_group_rewards(self, generated_paths, question, a_entity, expected_hops=2):
        """计算一组路径的奖励并标准化。"""
        rewards = [self.calculate_reward(path, question, a_entity, expected_hops) for path in generated_paths]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        if len(rewards) > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        return rewards, advantages
