import time
import os
import random
from openai import OpenAI
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']:
        num_tokens_limit = 128000
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002，gpt-3.5-turbo-1106']:
        num_tokens_limit = 4096
    else:
        # 对于未知模型（如 qwen 等），使用默认的 128000 token 限制
        num_tokens_limit = 128000
    return num_tokens_limit

PROMPT = """{instruction}

{input}"""

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--model_path', type=str, default='None')
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        parser.add_argument('--request_delay', type=float, help="delay between requests in seconds", default=0.5)
           
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.request_delay = getattr(args, 'request_delay', 0.5)  # 默认0.5秒延迟
        
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except KeyError:
            # 对于非 OpenAI 模型（如 qwen），使用简单估算：1 token ≈ 4 字符（英文）或 1.5 字符（中文）
            # 这里使用保守估计：字符数 / 2
            num_tokens = len(text) // 2
        return num_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        client_kwargs = {
            "api_key": os.environ.get('OPENAI_API_KEY'),
        }
        # 添加对base_url的支持，可以通过环境变量OPENAI_BASE_URL配置
        if 'OPENAI_BASE_URL' in os.environ:
            client_kwargs["base_url"] = os.environ['OPENAI_BASE_URL']
        client = OpenAI(**client_kwargs)
        self.client = client
    
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query
    
    def generate_sentence(self, llm_input):
        # 请求前增加随机延迟，避免多线程同时发送请求
        if self.request_delay > 0:
            delay = self.request_delay + random.uniform(0, 0.3)  # 增加随机性
            time.sleep(delay)

        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]

        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                result = response.choices[0].message.content.strip() # type: ignore
                return result
            except Exception as e:
                error_str = str(e)

                # 判断错误类型
                is_rate_limit = 'rate' in error_str.lower() or 'limit' in error_str.lower() or '429' in error_str
                is_timeout = 'timeout' in error_str.lower() or 'timed out' in error_str.lower()

                print(f"Request failed (attempt {cur_retry + 1}/{num_retry + 1}):")
                print("Message: ", llm_input[:200] + "..." if len(llm_input) > 200 else llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(f"Error: {error_str}")

                # 根据错误类型和重试次数，使用指数退避策略
                if is_rate_limit:
                    # 限流错误：使用更长的等待时间和指数退避
                    wait_time = 60 * (2 ** cur_retry) + random.uniform(0, 10)
                    print(f"Rate limit detected. Waiting {wait_time:.1f} seconds before retry...")
                elif is_timeout:
                    # 超时错误：使用中等等待时间
                    wait_time = 30 + random.uniform(0, 10)
                    print(f"Timeout detected. Waiting {wait_time:.1f} seconds before retry...")
                else:
                    # 其他错误：使用基础等待时间
                    wait_time = 30 + random.uniform(0, 5)
                    print(f"Other error. Waiting {wait_time:.1f} seconds before retry...")

                time.sleep(wait_time)
                cur_retry += 1
                continue

        print(f"Failed after {num_retry + 1} attempts. Returning None.")
        return None
