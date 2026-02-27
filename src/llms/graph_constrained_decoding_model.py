from src.graph_constrained_decoding import GraphConstrainedDecoding
from .base_hf_causal_model import HfCausalModel

class GraphConstrainedDecodingModel(HfCausalModel):
    def __init__(self, args):
        super().__init__(args)
    
    def generate_sentence(self, llm_input, trie, start_token_ids = None, end_token_ids = None, enable_constrained_by_default = True):
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, enable_constrained_by_default)
        try:
            res = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                generation_config=self.generation_cfg,
                prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(e)
            return None

        # 打印输入与模型原始输出（文本形式）
        print("=" * 80)
        print("【LLM输入】:")
        print(llm_input)
        print("=" * 80)
        print("【模型原始输出 - 包含特殊token】:")
        for idx, seq in enumerate(res.sequences):
            decoded_with_special = self.tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=False)
            print(f"Sequence {idx}: {repr(decoded_with_special)}")
        print("\n【处理后输出 - 去除特殊token】:")
        for idx, seq in enumerate(res.sequences):
            decoded_without_special = self.tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Sequence {idx}: {decoded_without_special}")
        print("=" * 80)

        response = []
        if len(res.sequences) == 1:
            return self.tokenizer.decode(res.sequences[0][input_ids.shape[1]:],skip_special_tokens=True)
        for r in res.sequences:
            response.append(self.tokenizer.decode(r[input_ids.shape[1]:], 
          skip_special_tokens=True))
        return response
        
