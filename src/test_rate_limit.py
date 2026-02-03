"""
测试API服务器的限流情况
支持自动获取模型列表并测试每个模型的限流状况
"""

import os
import time
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
from datetime import datetime
import dotenv

dotenv.load_dotenv()


class RateLimitTester:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.base_url = base_url
        self.results = defaultdict(lambda: {
            'success': 0,
            'failed': 0,
            'rate_limit_errors': 0,
            'timeout_errors': 0,
            'other_errors': 0,
            'response_times': [],
            'errors': []
        })

    def get_available_models(self):
        """获取服务器支持的模型列表"""
        print("正在获取可用模型列表...")
        try:
            models_response = self.client.models.list()
            models = [model.id for model in models_response.data]
            print(f"找到 {len(models)} 个可用模型:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            return models
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            print("将使用默认模型列表")
            return ['gpt-4o', 'gpt-4o-mini']

    def single_request(self, model_name, request_id):
        """发送单个请求"""
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, this is a test message."}],
                timeout=30,
                temperature=0.0,
                max_tokens=10
            )
            elapsed_time = time.time() - start_time
            return {
                'model': model_name,
                'request_id': request_id,
                'status': 'success',
                'response_time': elapsed_time,
                'error': None
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_str = str(e)
            error_type = 'other'

            # 判断错误类型
            if 'rate' in error_str.lower() or 'limit' in error_str.lower() or '429' in error_str:
                error_type = 'rate_limit'
            elif 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
                error_type = 'timeout'

            return {
                'model': model_name,
                'request_id': request_id,
                'status': 'failed',
                'response_time': elapsed_time,
                'error': error_str,
                'error_type': error_type
            }

    def test_model(self, model_name, num_requests, num_threads, delay=0):
        """测试单个模型"""
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name}")
        print(f"请求数量: {num_requests}, 并发线程: {num_threads}, 请求间隔: {delay}秒")
        print(f"{'='*60}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_requests):
                if delay > 0 and i > 0:
                    time.sleep(delay)
                future = executor.submit(self.single_request, model_name, i)
                futures.append(future)

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                model = result['model']
                completed += 1

                if result['status'] == 'success':
                    self.results[model]['success'] += 1
                    self.results[model]['response_times'].append(result['response_time'])
                    print(f"✓ [{completed}/{num_requests}] 请求 {result['request_id']} 成功 (耗时: {result['response_time']:.2f}s)")
                else:
                    self.results[model]['failed'] += 1
                    error_type = result['error_type']

                    if error_type == 'rate_limit':
                        self.results[model]['rate_limit_errors'] += 1
                        print(f"✗ [{completed}/{num_requests}] 请求 {result['request_id']} 失败 [限流] - {result['error'][:100]}")
                    elif error_type == 'timeout':
                        self.results[model]['timeout_errors'] += 1
                        print(f"✗ [{completed}/{num_requests}] 请求 {result['request_id']} 失败 [超时] - {result['error'][:100]}")
                    else:
                        self.results[model]['other_errors'] += 1
                        print(f"✗ [{completed}/{num_requests}] 请求 {result['request_id']} 失败 [其他] - {result['error'][:100]}")

                    self.results[model]['errors'].append({
                        'request_id': result['request_id'],
                        'error': result['error'],
                        'error_type': error_type
                    })

    def print_summary(self):
        """打印测试总结"""
        print(f"\n{'='*60}")
        print("测试总结")
        print(f"{'='*60}\n")

        for model, stats in self.results.items():
            total = stats['success'] + stats['failed']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            avg_response_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0

            print(f"模型: {model}")
            print(f"  总请求数: {total}")
            print(f"  成功: {stats['success']} ({success_rate:.1f}%)")
            print(f"  失败: {stats['failed']}")
            print(f"    - 限流错误: {stats['rate_limit_errors']}")
            print(f"    - 超时错误: {stats['timeout_errors']}")
            print(f"    - 其他错误: {stats['other_errors']}")
            print(f"  平均响应时间: {avg_response_time:.2f}秒")
            if stats['response_times']:
                print(f"  最快响应: {min(stats['response_times']):.2f}秒")
                print(f"  最慢响应: {max(stats['response_times']):.2f}秒")

            # 限流建议
            if stats['rate_limit_errors'] > 0:
                rate_limit_rate = stats['rate_limit_errors'] / total * 100
                if rate_limit_rate > 50:
                    print(f"  ⚠️  建议: 限流严重({rate_limit_rate:.1f}%)，建议减少并发数或增加请求间隔")
                elif rate_limit_rate > 20:
                    print(f"  ⚠️  建议: 有限流情况({rate_limit_rate:.1f}%)，建议适当降低并发数")
            print()

    def save_results(self, output_file):
        """保存结果到JSON文件"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'models': {}
        }

        for model, stats in self.results.items():
            total = stats['success'] + stats['failed']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            avg_response_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0

            output_data['models'][model] = {
                'total_requests': total,
                'success': stats['success'],
                'failed': stats['failed'],
                'success_rate': success_rate,
                'rate_limit_errors': stats['rate_limit_errors'],
                'timeout_errors': stats['timeout_errors'],
                'other_errors': stats['other_errors'],
                'avg_response_time': avg_response_time,
                'min_response_time': min(stats['response_times']) if stats['response_times'] else None,
                'max_response_time': max(stats['response_times']) if stats['response_times'] else None,
                'errors': stats['errors'][:10]  # 只保存前10个错误
            }

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='测试API服务器的限流情况')
    parser.add_argument('--base_url', type=str, default=None, help='API Base URL')
    parser.add_argument('--api_key', type=str, default=None, help='API Key')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='要测试的模型列表（不指定则自动获取所有模型）')
    parser.add_argument('--num_requests', type=int, default=20,
                        help='每个模型的请求数量')
    parser.add_argument('--num_threads', type=int, default=5,
                        help='并发线程数')
    parser.add_argument('--delay', type=float, default=0,
                        help='请求之间的延迟（秒）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--model_delay', type=float, default=2,
                        help='测试不同模型之间的延迟（秒）')

    args = parser.parse_args()

    # 从环境变量获取配置
    base_url = args.base_url or os.environ.get('OPENAI_BASE_URL')
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

    if not base_url or not api_key:
        print("错误: 请设置 OPENAI_BASE_URL 和 OPENAI_API_KEY 环境变量，或通过命令行参数提供")
        return

    print(f"API Base URL: {base_url}")
    print(f"并发线程数: {args.num_threads}")
    print(f"每个模型请求数: {args.num_requests}")
    print()

    tester = RateLimitTester(base_url, api_key)

    # 获取模型列表
    if args.models:
        models = args.models
        print(f"使用指定的模型列表: {', '.join(models)}\n")
    else:
        models = tester.get_available_models()
        print()

    if not models:
        print("错误: 没有可用的模型")
        return

    # 测试每个模型
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 开始测试模型: {model}")
        tester.test_model(model, args.num_requests, args.num_threads, args.delay)

        # 模型之间间隔
        if i < len(models):
            print(f"\n等待 {args.model_delay} 秒后测试下一个模型...")
            time.sleep(args.model_delay)

    # 打印总结
    tester.print_summary()

    # 保存结果
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'results/rate_limit_test_{timestamp}.json'

    tester.save_results(output_file)


if __name__ == '__main__':
    main()
