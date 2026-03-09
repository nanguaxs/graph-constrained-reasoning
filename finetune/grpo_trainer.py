"""GRPO 训练器：群体相对策略优化"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteriaList
import sys
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
        """处理 chat 模型的提示格式（与推理时保持一致）"""
        # 检查是否为 chat 模型
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            PATH_START = "<PATH>"
            if query.endswith(PATH_START):
                # 将 <PATH> 从 user message 中移出，避免被 chat template 打断
                user_content = query[:-len(PATH_START)]
                chat_query = [{"role": "user", "content": user_content}]
                return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True) + PATH_START
            else:
                chat_query = [{"role": "user", "content": query}]
                return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)
        else:
            return query

    def generate_group_paths(self, input_query, trie, num_generations):
        """生成一组路径（带约束）"""
        # 应用 chat template（与推理时保持一致）
        formatted_query = self.prepare_model_prompt(input_query)

        inputs = self.tokenizer(formatted_query, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        start_token_ids = self.tokenizer.convert_tokens_to_ids("<PATH>")
        end_token_ids = self.tokenizer.convert_tokens_to_ids("</PATH>")
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, True)

        # 添加停止条件（与自定义模型保持一致）
        stopping_criteria = StoppingCriteriaList([
            PathEndStoppingCriteria(start_token_ids, end_token_ids)
        ])

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

        generated_texts = [
            self.tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]
        return generated_texts, outputs.sequences, input_ids.shape[1]

    def compute_log_probs(self, input_ids, generated_ids):
        """计算生成序列的对数概率（无约束）"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, generated_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum(dim=1)

    def train_step(self, batch):
        """单步训练"""
        total_loss = 0
        for sample in batch:
            # 1. Rollout: 生成一组路径（带约束）
            generated_texts, sequences, prompt_len = self.generate_group_paths(
                sample["input_query"], sample["trie"], self.config.num_generations
            )

            # 2. 计算奖励和优势值
            rewards, advantages = self.reward_calculator.calculate_group_rewards(
                generated_texts, sample["a_entity"]
            )

            # 3. Update: 计算策略梯度（无约束）
            full_sequences = sequences
            old_log_probs = self.compute_log_probs(full_sequences, full_sequences)

            # 前向传播计算新的 log probs
            outputs = self.model(input_ids=full_sequences, attention_mask=torch.ones_like(full_sequences))
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = torch.gather(log_probs, 2, full_sequences[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1)

            # KL 散度惩罚
            kl_div = new_log_probs - old_log_probs

            # GRPO 损失
            loss = -(advantages.to(self.device) * new_log_probs).mean() + self.config.kl_penalty_beta * kl_div.mean()
            total_loss += loss.item()

            # 反向传播
            loss.backward()

        # 梯度更新
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / len(batch)

    def train(self, train_loader, num_epochs):
        """训练循环"""
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                if batch is None:
                    continue
                loss = self.train_step(batch)
                epoch_loss += loss
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
