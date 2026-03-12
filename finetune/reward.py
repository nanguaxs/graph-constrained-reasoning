"""奖励函数：答案命中 + 真值路径匹配 + 动态绕路惩罚 + 语义辅助项。"""
import re
from typing import Sequence

import requests
import torch

from logging_utils import get_logger


logger = get_logger("reward")


class PathRewardCalculator:
    def __init__(
        self,
        embedding_api_url="http://localhost:8000/v1/embeddings",
        embedding_model_name="text-embedding-3-small",
        embedding_api_key="",
        use_semantic_reward=True,
    ):
        self.embedding_api_url = embedding_api_url
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.use_semantic_reward = use_semantic_reward
        self.embedding_cache = {}

    def get_embeddings(self, texts):
        """调用服务端 API 获取嵌入向量，并缓存重复文本。"""
        normalized_texts = [str(text).strip() for text in texts]
        uncached_texts = [text for text in dict.fromkeys(normalized_texts) if text not in self.embedding_cache]

        if uncached_texts:
            headers = {"Content-Type": "application/json"}
            if self.embedding_api_key:
                headers["Authorization"] = f"Bearer {self.embedding_api_key}"

            response = None
            try:
                response = requests.post(
                    self.embedding_api_url,
                    json={"input": uncached_texts, "model": self.embedding_model_name},
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.HTTPError as exc:
                status_code = response.status_code if response is not None else "unknown"
                response_preview = response.text[:500] if response is not None else ""
                raise RuntimeError(
                    f"embedding API 返回 HTTP 错误: url={self.embedding_api_url} model={self.embedding_model_name} "
                    f"status={status_code} body={response_preview}"
                ) from exc
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"embedding API 请求异常: url={self.embedding_api_url} model={self.embedding_model_name} error={exc}"
                ) from exc
            except ValueError as exc:
                response_preview = response.text[:500] if response is not None else ""
                raise RuntimeError(
                    f"embedding API 返回的不是合法 JSON: url={self.embedding_api_url} model={self.embedding_model_name} "
                    f"body={response_preview}"
                ) from exc

            if "data" in payload:
                embeddings = [item["embedding"] for item in payload["data"]]
            elif "embeddings" in payload:
                embeddings = payload["embeddings"]
            elif "embedding" in payload:
                raw_embedding = payload["embedding"]
                if isinstance(raw_embedding, list) and raw_embedding and isinstance(raw_embedding[0], (list, tuple)):
                    embeddings = raw_embedding
                else:
                    embeddings = [raw_embedding]
            else:
                raise RuntimeError(
                    f"embedding API 返回缺少 embeddings 字段: url={self.embedding_api_url} "
                    f"keys={sorted(payload.keys())}"
                )

            if len(embeddings) != len(uncached_texts):
                raise RuntimeError(
                    f"embedding 数量与请求文本数量不一致: expected={len(uncached_texts)} actual={len(embeddings)}"
                )

            for text, embedding in zip(uncached_texts, embeddings):
                self.embedding_cache[text] = torch.tensor(embedding, dtype=torch.float32)

        return torch.stack([self.embedding_cache[text] for text in normalized_texts])

    def parse_path_segments(self, path_text):
        """将路径文本解析为 [实体, 关系, 实体, ...] 的分段列表。"""
        # logger.debug("原始路径文本: %r", path_text)

        if not path_text:
            logger.debug("路径文本为空")
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
        # logger.debug("规范化路径文本: %r", normalized_text)
        # logger.debug("路径分段结果: %s", path_segments)
        return path_segments

    def build_path_info(self, path_text):
        """将路径文本解析为便于打分的结构信息。"""
        path_segments = self.parse_path_segments(path_text)
        if not path_segments:
            return None

        if len(path_segments) % 2 == 0:
            logger.debug("路径分段数为偶数，说明路径停在关系上")
            return None

        if len(path_segments) < 3:
            logger.debug("路径分段过短，无法构成完整三元组")
            return None

        triples = []
        for index in range(0, len(path_segments) - 2, 2):
            head = path_segments[index]
            relation = path_segments[index + 1]
            tail = path_segments[index + 2]
            triples.append((head, relation, tail))

        relations = [relation for _, relation, _ in triples]
        normalized_text = " -> ".join(path_segments)
        final_entity = triples[-1][2]
        path_info = {
            "text": normalized_text,
            "segments": path_segments,
            "triples": triples,
            "relations": relations,
            "hops": len(triples),
            "final_entity": final_entity,
        }
        # logger.debug("提取的终点实体: %r", final_entity)
        # logger.debug("解析出的三元组: %s", triples)
        return path_info

    def parse_ground_paths(self, ground_paths):
        """解析真值路径集合，忽略无效路径。"""
        if ground_paths is None:
            return []
        if isinstance(ground_paths, str):
            ground_paths = [ground_paths]

        parsed_ground_paths = []
        for path_text in ground_paths:
            path_info = self.build_path_info(path_text)
            if path_info is not None:
                parsed_ground_paths.append(path_info)
        return parsed_ground_paths

    @staticmethod
    def compute_prefix_length(left: Sequence[tuple], right: Sequence[tuple]) -> int:
        prefix_length = 0
        for left_item, right_item in zip(left, right):
            if left_item != right_item:
                break
            prefix_length += 1
        return prefix_length

    @staticmethod
    def compute_lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
        if not left or not right:
            return 0

        dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for left_index, left_value in enumerate(left, start=1):
            for right_index, right_value in enumerate(right, start=1):
                if left_value == right_value:
                    dp[left_index][right_index] = dp[left_index - 1][right_index - 1] + 1
                else:
                    dp[left_index][right_index] = max(
                        dp[left_index - 1][right_index],
                        dp[left_index][right_index - 1],
                    )
        return dp[-1][-1]

    def compute_match_components(self, generated_info, ground_info):
        """计算生成路径与某条真值路径的结构匹配分。"""
        exact = float(generated_info["triples"] == ground_info["triples"])

        prefix_length = self.compute_prefix_length(
            generated_info["triples"],
            ground_info["triples"],
        )
        prefix_ratio = prefix_length / ground_info["hops"] if ground_info["hops"] > 0 else 0.0

        lcs_length = self.compute_lcs_length(
            generated_info["relations"],
            ground_info["relations"],
        )
        rel_sim = lcs_length / len(ground_info["relations"]) if ground_info["relations"] else 0.0

        match_score = 2.5 * exact + 1.75 * prefix_ratio + 1.0 * rel_sim
        return {
            "exact": exact,
            "prefix_ratio": prefix_ratio,
            "rel_sim": rel_sim,
            "match_score": match_score,
        }

    def select_reference_ground_path(self, generated_info, ground_infos):
        """在真值路径集合中选出与生成路径最接近的参照路径。"""
        best_ground_info = None
        best_components = None

        for ground_info in ground_infos:
            components = self.compute_match_components(generated_info, ground_info)
            if best_components is None or components["match_score"] > best_components["match_score"]:
                best_ground_info = ground_info
                best_components = components

        return best_ground_info, best_components

    def compute_semantic_reward(self, generated_info, ground_infos):
        """计算生成路径与真值路径集合的弱语义对齐分。"""
        if not ground_infos:
            return 0.0, None

        ground_texts = [ground_info["text"] for ground_info in ground_infos]
        try:
            generated_emb = self.get_embeddings([generated_info["text"]])
            ground_embs = self.get_embeddings(ground_texts)
        except Exception:
            logger.exception(
                "语义辅助项计算失败，已退化为 0: url=%s model=%s generated=%r ground_count=%s",
                self.embedding_api_url,
                self.embedding_model_name,
                generated_info["text"],
                len(ground_texts),
            )
            return 0.0, None

        similarities = torch.cosine_similarity(generated_emb, ground_embs, dim=1)
        best_index = similarities.argmax().item()
        best_similarity = similarities[best_index].item()
        semantic_reward = 0.5 * best_similarity
        return semantic_reward, {
            "best_similarity": best_similarity,
            "best_ground_text": ground_texts[best_index],
        }

    def extract_final_entity(self, path_text):
        """从路径文本中提取终点实体。"""
        path_info = self.build_path_info(path_text)
        if path_info is None:
            logger.debug("未解析出有效路径，终点实体为 None")
            return None

        final_entity = path_info["final_entity"]
        # logger.debug("提取的终点实体: %r", final_entity)
        return final_entity

    def count_hops(self, path_text):
        """根据解析后的路径结构统计 hop 数。"""
        path_info = self.build_path_info(path_text)
        if path_info is None:
            return 0
        return path_info["hops"]

    def calculate_reward(self, generated_path, question, a_entity, ground_paths):
        """计算单条路径的奖励。"""
        logger.debug("开始奖励计算: path=%s | question=%s", generated_path, question)

        generated_info = self.build_path_info(generated_path)
        if generated_info is None:
            logger.debug("未解析出有效路径，返回 -1.0")
            return -1.0

        if isinstance(a_entity, str):
            a_entity = [a_entity]
        answer_entities = {str(answer).strip() for answer in a_entity if str(answer).strip()}

        parsed_ground_paths = self.parse_ground_paths(ground_paths)
        answer_reward = 5.0 if generated_info["final_entity"] in answer_entities else 0.0
        logger.debug("答案奖励: %.4f", answer_reward)

        entities = generated_info["segments"][::2]
        seen_entities = set()
        revisit_count = 0
        for entity in entities:
            if entity in seen_entities:
                revisit_count += 1
            else:
                seen_entities.add(entity)
        loop_penalty = max(-4.0, -2.0 * revisit_count)
        logger.debug("绕圈惩罚: revisits=%s loop_penalty=%.4f", revisit_count, loop_penalty)

        reference_ground_info, match_components = self.select_reference_ground_path(
            generated_info,
            parsed_ground_paths,
        )
        if match_components is None:
            path_match_reward = 0.0
            detour_reward = 0.0
            logger.debug("无可用真值路径，结构匹配分与绕路惩罚均记为 0")
        else:
            path_match_reward = match_components["match_score"]
            extra_hops = max(0, generated_info["hops"] - reference_ground_info["hops"])
            detour_reward = -1.2 * extra_hops
            # logger.debug(
            #     "结构匹配: ref=%s exact=%.2f prefix_ratio=%.2f rel_sim=%.2f path_match=%.4f detour=%.4f",
            #     reference_ground_info["text"],
            #     match_components["exact"],
            #     match_components["prefix_ratio"],
            #     match_components["rel_sim"],
            #     path_match_reward,
            #     detour_reward,
            # )

        if self.use_semantic_reward:
            semantic_reward, semantic_details = self.compute_semantic_reward(
                generated_info,
                parsed_ground_paths,
            )
            if semantic_details is None:
                logger.debug("语义辅助项: 0.0000")
        else:
            semantic_reward = 0.0
            semantic_details = None
            logger.debug("语义辅助项已关闭，按 0.0000 处理")
        # else:
            # logger.debug(
            #     "语义辅助项: ref=%s similarity=%.4f semantic=%.4f",
            #     semantic_details["best_ground_text"],
            #     semantic_details["best_similarity"],
            #     semantic_reward,
            # )

        reward = answer_reward + path_match_reward + detour_reward + loop_penalty + semantic_reward
        logger.debug("最终奖励: %.4f", reward)
        return reward

    def calculate_group_rewards(self, generated_paths, question, a_entity, ground_paths):
        """计算一组路径的奖励并标准化。"""
        rewards = [
            self.calculate_reward(path, question, a_entity, ground_paths)
            for path in generated_paths
        ]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        if len(rewards) > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        return rewards, advantages
