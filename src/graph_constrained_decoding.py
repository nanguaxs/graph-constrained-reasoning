from typing import List
import torch

class GraphConstrainedDecoding:
    def __init__(self, tokenizer, trie, start_token_ids = None, end_token_ids = None, enable_constrained_by_default = False):
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token = start_token_ids
        self.end_token = end_token_ids
        self.all_tokens = list(range(len(tokenizer)))
        self.constrained_flag = enable_constrained_by_default
        self.L_input = None
        self.completed = False

    def check_constrained_flag(self, sent: torch.Tensor):
        # Check start
        matched_start_token = torch.where(sent == self.start_token)[0]
        if len(matched_start_token) == 0:
            return False, len(sent)
        last_start_tokens = torch.where(sent == self.start_token)[0][-1]
        end_token_number = len(torch.where(sent[last_start_tokens:] == self.end_token)[0])
        # GCR not closed
        if end_token_number == 0:
            self.last_start_token = last_start_tokens
            return True, last_start_tokens
        else:
            self.last_start_token = None
            self.completed = True
            return False, len(sent)
    
    def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
        if self.completed:
            return [self.tokenizer.eos_token_id]
        constrained_flag = self.constrained_flag
        # Check if enter the constrained decoding
        if self.start_token is not None and self.end_token is not None:
            constrained_flag, L_input = self.check_constrained_flag(sent)
        # Assign self.L_input with the input length
        else:
            if self.L_input is None:
                self.L_input = len(sent)
            L_input = self.L_input
        allow_tokens = self.all_tokens
        if constrained_flag:
            allow_tokens = self.trie.get(sent.tolist()[L_input:])
            if len(allow_tokens) == 0:
                return self.all_tokens
        return allow_tokens

    # def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
    #     constrained_flag = self.constrained_flag
    #     # 调试：每次调用时打印 start_token 和 end_token 的值（仅第一次）
    #     if self.L_input is None:
    #         print(f"[调试-首次调用] start_token={self.start_token}, end_token={self.end_token}, constrained_flag={constrained_flag}")
    #     # Check if enter the constrained decoding
    #     if self.start_token is not None and self.end_token is not None:
    #         # 调试：仅在第一次调用时打印 start_token 是否出现在 sent 中
    #         if self.L_input is None:
    #             print(f"[调试] start_token={self.start_token}, sent 中是否含有: {self.start_token in sent.tolist()}, sent 末尾5个token={sent.tolist()[-5:]}")
    #         constrained_flag, L_input = self.check_constrained_flag(sent)
    #     # Assign self.L_input with the input length
    #     else:
    #         if self.L_input is None:
    #             self.L_input = len(sent)
    #         L_input = self.L_input

    #     allow_tokens = self.all_tokens
    #     if constrained_flag:
    #         generated = sent.tolist()[L_input:]
    #         # 调试：约束首次激活时打印（generated 只含 <PATH> 一个 token）
    #         if len(generated) == 1:
    #             print(f"[约束激活] L_input={L_input}, 前缀token={generated}, 解码='{self.tokenizer.decode(generated)}'")
    #         allow_tokens = self.trie.get(generated)
    #         if len(allow_tokens) == 0:
    #             # 调试：fallback 触发时打印当前前缀
    #             print(f"[Fallback] trie 无匹配，当前前缀='{self.tokenizer.decode(generated)}'，退回无约束")
    #             return self.all_tokens
    #     return allow_tokens