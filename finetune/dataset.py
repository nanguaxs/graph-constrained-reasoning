"""图约束路径生成数据集。"""
import os
import sys

from datasets import load_dataset
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
