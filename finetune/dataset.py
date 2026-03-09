"""图约束路径生成数据集"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import sys
sys.path.append('..')
from src import utils
from src.qa_prompt_builder import ChinesePathGenerationWithAnswerPromptBuilder

class PathGenerationDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, index_path_length=2, undirected=False):
        self.dataset = load_dataset(data_path, split=split)
        self.tokenizer = tokenizer
        self.prompt_builder = ChinesePathGenerationWithAnswerPromptBuilder(
            tokenizer, "zero-shot", index_path_length, undirected, add_rule=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # 构建输入提示
        input_query, ground_paths, trie = self.prompt_builder.process_input(data)

        if trie is None:
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
        """过滤掉 None 样本"""
        batch = [b for b in batch if b is not None]
        return batch if len(batch) > 0 else None
