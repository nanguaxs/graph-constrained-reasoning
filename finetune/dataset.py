"""图约束路径生成数据集。"""
import os
import sys

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from logging_utils import get_logger


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.qa_prompt_builder import ChinesePathGenerationWithAnswerPromptBuilder


logger = get_logger("dataset")


class PathGenerationDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, index_path_length=2, undirected=False):
        if data_path.endswith(".json"):
            self.dataset = load_dataset("json", data_files=data_path, split=split)
        else:
            self.dataset = load_dataset(data_path, split=split)

        self.tokenizer = tokenizer
        self.prompt_builder = ChinesePathGenerationWithAnswerPromptBuilder(
            tokenizer,
            "zero-shot",
            undirected=undirected,
            index_path_length=index_path_length,
            add_rule=False,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_query, ground_paths, trie = self.prompt_builder.process_input(data)

        if trie is None:
            logger.debug("idx=%s, id=%s 的 trie 为 None，样本将被跳过", idx, data.get("id", "unknown"))
            return None

        return {
            "id": data["id"],
            "question": data["question"],
            "q_entity": data["q_entity"],
            "a_entity": data["a_entity"],
            "graph": data["graph"],
            "input_query": input_query,
            "ground_paths": ground_paths,
            "trie": trie,
        }

    def collate_fn(self, batch):
        """过滤掉无效样本。"""
        original_len = len(batch)
        batch = [sample for sample in batch if sample is not None]
        if len(batch) == 0:
            logger.debug("当前 batch 全部无效（原始 %s 条），返回 None", original_len)
            return None
        if len(batch) != original_len:
            logger.debug("当前 batch 过滤后剩余 %s/%s 条有效样本", len(batch), original_len)
        return batch


class SFTPathDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        tokenizer,
        index_path_length=2,
        undirected=False,
        max_ground_paths_per_sample=None,
    ):
        if data_path.endswith(".json"):
            self.dataset = load_dataset("json", data_files=data_path, split=split)
        else:
            self.dataset = load_dataset(data_path, split=split)

        self.tokenizer = tokenizer
        self.max_ground_paths_per_sample = max_ground_paths_per_sample
        self.prompt_builder = ChinesePathGenerationWithAnswerPromptBuilder(
            tokenizer,
            "zero-shot",
            undirected=undirected,
            index_path_length=index_path_length,
            add_rule=False,
        )
        self.samples = []
        self._build_samples()

    @staticmethod
    def _strip_path_tags(path_text):
        normalized_text = str(path_text).strip()
        if "<PATH>" in normalized_text:
            normalized_text = normalized_text.split("<PATH>", 1)[-1]
        if "</PATH>" in normalized_text:
            normalized_text = normalized_text.split("</PATH>", 1)[0]
        return normalized_text.strip()

    @staticmethod
    def _canonicalize_path(path_text):
        return " ".join(str(path_text).split()).strip()

    def _prepare_model_prompt(self, query):
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            path_start = "<PATH>"
            if query.endswith(path_start):
                user_content = query[:-len(path_start)]
                chat_query = [{"role": "user", "content": user_content}]
                return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True) + path_start

            chat_query = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)

        return query

    def _build_target_text(self, prompt_text, clean_path_text):
        full_path_text = f"<PATH>{clean_path_text}</PATH>"
        if prompt_text.endswith("<PATH>"):
            return f"{clean_path_text}</PATH>"
        return full_path_text

    def _select_ground_paths(self, ground_paths):
        if ground_paths is None:
            return []
        if isinstance(ground_paths, str):
            ground_paths = [ground_paths]

        selected_paths = []
        seen_paths = set()
        for path_text in ground_paths:
            clean_path = self._strip_path_tags(path_text)
            canonical_path = self._canonicalize_path(clean_path)
            if not canonical_path or canonical_path in seen_paths:
                continue

            seen_paths.add(canonical_path)
            selected_paths.append(clean_path)
            if (
                self.max_ground_paths_per_sample is not None
                and len(selected_paths) >= self.max_ground_paths_per_sample
            ):
                break

        return selected_paths

    def _build_samples(self):
        skipped_samples = 0
        for data in self.dataset:
            input_query, ground_paths, _ = self.prompt_builder.process_input(data, return_tire=False)
            selected_paths = self._select_ground_paths(ground_paths)
            if len(selected_paths) == 0:
                skipped_samples += 1
                continue

            prompt_text = self._prepare_model_prompt(input_query)
            source_id = data.get("id", f"sample_{len(self.samples)}")
            for path_index, clean_path in enumerate(selected_paths):
                self.samples.append(
                    {
                        "id": f"{source_id}__path_{path_index}",
                        "source_id": source_id,
                        "prompt_text": prompt_text,
                        "target_text": self._build_target_text(prompt_text, clean_path),
                        "ground_path": clean_path,
                    }
                )

        logger.info(
            "SFT 数据集展开完成: raw_samples=%s sft_samples=%s skipped_without_gt=%s max_ground_paths_per_sample=%s",
            len(self.dataset),
            len(self.samples),
            skipped_samples,
            self.max_ground_paths_per_sample,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        if len(batch) == 0:
            return None

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("SFT 训练要求 tokenizer.eos_token_id 存在")

        input_id_tensors = []
        label_tensors = []
        sequence_lengths = []

        for sample in batch:
            prompt_ids = self.tokenizer(
                sample["prompt_text"],
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
            target_ids = self.tokenizer(
                sample["target_text"],
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids

            full_input_ids = prompt_ids + target_ids + [eos_token_id]
            full_labels = ([-100] * len(prompt_ids)) + target_ids + [eos_token_id]

            input_id_tensors.append(torch.tensor(full_input_ids, dtype=torch.long))
            label_tensors.append(torch.tensor(full_labels, dtype=torch.long))
            sequence_lengths.append(len(full_input_ids))

        input_ids = pad_sequence(
            input_id_tensors,
            batch_first=True,
            padding_value=eos_token_id,
        )
        labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(sequence_lengths):
            attention_mask[index, :length] = 1

        return {
            "sample_ids": [sample["id"] for sample in batch],
            "source_ids": [sample["source_id"] for sample in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
