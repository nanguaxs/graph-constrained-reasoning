"""GRPO 训练器：群体相对策略优化"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import StoppingCriteriaList

sys.path.append('..')
from src.graph_constrained_decoding import GraphConstrainedDecoding, PathEndStoppingCriteria


class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_calculator, optimizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.optimizer = optimizer
        self.config = config
        self.device = model.device

    def prepare_model_prompt(self, query):
        """处理 chat 模型的提示格式（与推理时保持一致）。"""
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

    def generate_paths_once(self, input_query, trie, num_generations):
        """单轮采样一组路径。"""
        formatted_query = self.prepare_model_prompt(input_query)
        inputs = self.tokenizer(formatted_query, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        start_token_ids = self.tokenizer.convert_tokens_to_ids("<PATH>")
        end_token_ids = self.tokenizer.convert_tokens_to_ids("</PATH>")
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, True)
        stopping_criteria = StoppingCriteriaList([PathEndStoppingCriteria(start_token_ids, end_token_ids)])

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=num_generations,
            do_sample=True,
            temperature=self.config.temperature,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prompt_len = input_ids.shape[1]
        generated_texts = [
            self.tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
            for sequence in outputs.sequences
        ]
        sequence_lengths = [self.get_sequence_length(sequence, prompt_len) for sequence in outputs.sequences]
        return generated_texts, outputs.sequences, sequence_lengths

    def generate_group_paths(self, input_query, trie, num_generations):
        """去重采样路径；不足 K 条时继续补采样。"""
        unique_texts = []
        unique_sequences = []
        unique_lengths = []
        seen_paths = set()
        stagnant_rounds = 0

        for round_index in range(self.config.max_resample_rounds):
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

            print(
                f"[DEBUG][generate] round={round_index + 1}, sampled={len(round_texts)}, "
                f"new_unique={new_unique_count}, total_unique={len(unique_texts)}/{num_generations}"
            )

            if new_unique_count == 0:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0

            if stagnant_rounds >= 2:
                print("[DEBUG][generate] 连续两轮没有新增唯一路径，提前停止补采样")
                break

        if len(unique_texts) < num_generations:
            print(f"[DEBUG][generate] 仅获得 {len(unique_texts)}/{num_generations} 条唯一路径，将使用现有路径训练")

        if not unique_sequences:
            return [], torch.empty((0, 0), dtype=torch.long, device=self.device), torch.empty((0, 0), dtype=torch.long, device=self.device)

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
        total_loss = 0
        print(f"[DEBUG][train_step] 本次 batch 样本数: {len(batch)}")

        for sample in batch:
            print(f"[DEBUG][train_step] 开始处理样本 id={sample.get('id', 'unknown')}")
            generated_texts, sequences, sequence_attention_mask = self.generate_group_paths(
                sample["input_query"], sample["trie"], self.config.num_generations
            )

            if not generated_texts:
                print(f"[DEBUG][train_step] 样本 id={sample.get('id', 'unknown')} 未生成有效路径，已跳过")
                continue

            print(f"[DEBUG][train_step] 去重后路径数: {len(generated_texts)}")
            rewards, advantages = self.reward_calculator.calculate_group_rewards(
                generated_texts,
                sample["question"],
                sample["a_entity"],
                sample["ground_paths"],
            )
            print(f"[DEBUG][train_step] rewards={rewards.tolist()}, advantages={advantages.tolist()}")

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
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / len(batch)

    def train(self, train_loader, num_epochs):
        """训练循环。"""
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            skipped_batches = 0
            total_batches = len(train_loader)
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                if batch is None:
                    skipped_batches += 1
                    print(f"[DEBUG][train] epoch={epoch+1} 遇到空 batch，已跳过")
                    continue
                loss = self.train_step(batch)
                epoch_loss += loss

            effective_batches = total_batches - skipped_batches
            print(f"[DEBUG][train] epoch={epoch+1} 总 batch={total_batches}, 跳过={skipped_batches}, 有效={effective_batches}")
            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
