"""GRPO 训练器：群体相对策略优化。"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import StoppingCriteriaList

from logging_utils import get_logger

sys.path.append('..')
from src.graph_constrained_decoding import GraphConstrainedDecoding, PathEndStoppingCriteria


logger = get_logger("trainer")


class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_calculator, optimizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.optimizer = optimizer
        self.config = config
        self.device = model.device
        self._warned_num_beams = False

    def prepare_model_prompt(self, query):
        """处理 chat 模型的提示格式，与推理阶段保持一致。"""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            path_start = "<PATH>"
            if query.endswith(path_start):
                user_content = query[:-len(path_start)]
                chat_query = [{"role": "user", "content": user_content}]
                return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True) + path_start

            chat_query = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)

        return query

    @staticmethod
    def canonicalize_generated_path(path_text):
        return " ".join(str(path_text).split()).strip()

    def get_sequence_length(self, sequence, prompt_len):
        eos_token_id = self.tokenizer.eos_token_id
        generated_tokens = sequence[prompt_len:]
        eos_positions = (generated_tokens == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) == 0:
            return sequence.shape[0]
        return prompt_len + eos_positions[0].item() + 1

    def build_attention_mask(self, padded_sequences, sequence_lengths):
        attention_mask = torch.zeros_like(padded_sequences)
        for index, length in enumerate(sequence_lengths):
            attention_mask[index, :length] = 1
        return attention_mask

    def _resolve_num_beams(self, num_generations):
        effective_num_beams = max(self.config.num_beams, num_generations)
        if effective_num_beams != self.config.num_beams and not self._warned_num_beams:
            logger.warning(
                "generation_mode=%s 时 num_beams=%s 小于 num_generations=%s，已自动提升到 %s",
                self.config.generation_mode,
                self.config.num_beams,
                num_generations,
                effective_num_beams,
            )
            self._warned_num_beams = True
        return effective_num_beams

    def _build_generation_kwargs(self, gcr, stopping_criteria, num_generations):
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_return_sequences": num_generations,
            "prefix_allowed_tokens_fn": gcr.allowed_tokens_fn,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if self.config.generation_mode == "sampling":
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                }
            )
        elif self.config.generation_mode == "beam_search":
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": self._resolve_num_beams(num_generations),
                    "early_stopping": self.config.early_stopping,
                }
            )
        elif self.config.generation_mode == "beam_sample":
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "num_beams": self._resolve_num_beams(num_generations),
                    "temperature": self.config.temperature,
                    "early_stopping": self.config.early_stopping,
                }
            )
        else:
            raise ValueError(f"不支持的 generation_mode: {self.config.generation_mode}")

        if generation_kwargs.get("do_sample"):
            generation_kwargs["top_p"] = self.config.top_p
            if self.config.top_k > 0:
                generation_kwargs["top_k"] = self.config.top_k

        return generation_kwargs

    def generate_paths_once(self, input_query, trie, num_generations):
        """按配置的生成模式单轮生成一组路径。"""
        formatted_query = self.prepare_model_prompt(input_query)
        inputs = self.tokenizer(formatted_query, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        start_token_ids = self.tokenizer.convert_tokens_to_ids("<PATH>")
        end_token_ids = self.tokenizer.convert_tokens_to_ids("</PATH>")
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, True)
        stopping_criteria = StoppingCriteriaList([PathEndStoppingCriteria(start_token_ids, end_token_ids)])
        generation_kwargs = self._build_generation_kwargs(gcr, stopping_criteria, num_generations)

        logger.debug(
            "生成参数: mode=%s, num_return_sequences=%s, num_beams=%s, temperature=%.3f, top_p=%.3f, top_k=%s",
            self.config.generation_mode,
            num_generations,
            generation_kwargs.get("num_beams", 1),
            generation_kwargs.get("temperature", 1.0),
            generation_kwargs.get("top_p", 1.0),
            generation_kwargs.get("top_k", 0),
        )

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        prompt_len = input_ids.shape[1]
        generated_texts = [
            self.tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
            for sequence in outputs.sequences
        ]
        sequence_lengths = [self.get_sequence_length(sequence, prompt_len) for sequence in outputs.sequences]
        return generated_texts, outputs.sequences, sequence_lengths

    def generate_group_paths(self, input_query, trie, num_generations):
        """去重生成路径；不足 K 条时继续补生成。"""
        unique_texts = []
        unique_sequences = []
        unique_lengths = []
        seen_paths = set()
        stagnant_rounds = 0
        max_rounds = 1 if self.config.generation_mode == "beam_search" else self.config.max_resample_rounds

        for round_index in range(max_rounds):
            remaining = num_generations - len(unique_texts)
            if remaining <= 0:
                break

            round_texts, round_sequences, round_lengths = self.generate_paths_once(input_query, trie, remaining)
            new_unique_count = 0

            for text, sequence, length in zip(round_texts, round_sequences, round_lengths):
                canonical_path = self.canonicalize_generated_path(text)
                if canonical_path in seen_paths:
                    continue

                seen_paths.add(canonical_path)
                unique_texts.append(text)
                unique_sequences.append(sequence[:length])
                unique_lengths.append(length)
                new_unique_count += 1

                if len(unique_texts) >= num_generations:
                    break

            logger.debug(
                "生成轮次: round=%s, sampled=%s, new_unique=%s, total_unique=%s/%s",
                round_index + 1,
                len(round_texts),
                new_unique_count,
                len(unique_texts),
                num_generations,
            )

            stagnant_rounds = stagnant_rounds + 1 if new_unique_count == 0 else 0
            if stagnant_rounds >= 2:
                logger.debug("连续两轮没有新增唯一路径，提前停止补生成")
                break

        if len(unique_texts) < num_generations:
            logger.warning(
                "仅获得 %s/%s 条唯一路径，将使用现有路径继续训练",
                len(unique_texts),
                num_generations,
            )

        if not unique_sequences:
            empty_tensor = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return [], empty_tensor, empty_tensor

        padded_sequences = pad_sequence(
            unique_sequences,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id,
        )
        attention_mask = self.build_attention_mask(padded_sequences, unique_lengths)
        return unique_texts, padded_sequences, attention_mask

    def compute_log_probs(self, input_ids, attention_mask):
        """计算生成序列的对数概率。"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            token_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
            return (token_log_probs * token_mask).sum(dim=1)

    def train_step(self, batch):
        """单步训练。"""
        total_loss = 0.0
        effective_samples = 0
        logger.debug("本次 batch 样本数: %s", len(batch))

        for sample in batch:
            sample_id = sample.get("id", "unknown")
            logger.debug("开始处理样本 id=%s", sample_id)

            generated_texts, sequences, sequence_attention_mask = self.generate_group_paths(
                sample["input_query"],
                sample["trie"],
                self.config.num_generations,
            )
            if len(generated_texts) == 0:
                logger.warning("样本 id=%s 未生成有效路径，已跳过", sample_id)
                continue

            rewards, advantages = self.reward_calculator.calculate_group_rewards(
                generated_texts,
                sample["question"],
                sample["a_entity"],
                sample["ground_paths"],
            )
            logger.debug(
                "样本 id=%s rewards=%s advantages=%s",
                sample_id,
                rewards.tolist(),
                advantages.tolist(),
            )
            logger.info(
                "sample_id=%s mode=%s unique=%s/%s reward_max=%.4f reward_mean=%.4f",
                sample_id,
                self.config.generation_mode,
                len(generated_texts),
                self.config.num_generations,
                rewards.max().item(),
                rewards.mean().item(),
            )

            full_sequences = sequences
            old_log_probs = self.compute_log_probs(full_sequences, sequence_attention_mask)

            outputs = self.model(input_ids=full_sequences, attention_mask=sequence_attention_mask)
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, full_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
            token_mask = sequence_attention_mask[:, 1:].to(token_log_probs.dtype)
            new_log_probs = (token_log_probs * token_mask).sum(dim=1)

            kl_div = new_log_probs - old_log_probs
            loss = -(advantages.to(self.device) * new_log_probs).mean() + self.config.kl_penalty_beta * kl_div.mean()
            total_loss += loss.item()
            effective_samples += 1
            loss.backward()

        if effective_samples == 0:
            self.optimizer.zero_grad()
            return 0.0

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / effective_samples

    def train(self, train_loader, num_epochs):
        """训练循环。"""
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            skipped_batches = 0
            effective_batches = 0
            total_batches = len(train_loader)

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                if batch is None:
                    skipped_batches += 1
                    logger.debug("epoch=%s 遇到空 batch，已跳过", epoch + 1)
                    continue

                loss = self.train_step(batch)
                epoch_loss += loss
                effective_batches += 1

            avg_loss = epoch_loss / effective_batches if effective_batches > 0 else 0.0
            logger.info(
                "epoch=%s avg_loss=%.4f total_batches=%s skipped=%s effective=%s",
                epoch + 1,
                avg_loss,
                total_batches,
                skipped_batches,
                effective_batches,
            )
