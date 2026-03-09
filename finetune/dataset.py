"""图约束路径生成数据集"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import utils
from src.qa_prompt_builder import ChinesePathGenerationWithAnswerPromptBuilder

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
        # 构建输入提示
        input_query, ground_paths, trie = self.prompt_builder.process_input(data)

        if trie is None:
            print(f"[DEBUG][dataset] idx={idx}, id={data.get('id', 'unknown')} 的 trie 为 None，样本将被跳过")
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
        original_len = len(batch)
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            print(f"[DEBUG][dataloader] 当前 batch 全部无效（原始 {original_len} 条），返回 None")
            return None
        if len(batch) != original_len:
            print(f"[DEBUG][dataloader] 当前 batch 过滤后剩余 {len(batch)}/{original_len} 条有效样本")
        return batch
