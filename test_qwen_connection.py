#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Qwen 模型连接的简单脚本
"""

import os
from openai import OpenAI
import dotenv

# 加载环境变量
dotenv.load_dotenv()

def test_qwen_connection():
    """测试 Qwen 模型连接"""

    print("=" * 60)
    print("Qwen 模型连接测试")
    print("=" * 60)

    # 1. 检查环境变量
    print("\n[步骤 1] 检查环境变量配置...")
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL')

    if not api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY 环境变量")
        print("   请在 .env 文件中设置: OPENAI_API_KEY=your_api_key")
        return False
    else:
        print(f"✅ OPENAI_API_KEY: {api_key[:10]}...{api_key[-4:]}")

    if not base_url:
        print("⚠️  警告: 未设置 OPENAI_BASE_URL，将使用默认 OpenAI 端点")
    else:
        print(f"✅ OPENAI_BASE_URL: {base_url}")

    # 2. 初始化客户端
    print("\n[步骤 2] 初始化 OpenAI 客户端...")
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return False

    # 3. 测试模型调用
    print("\n[步骤 3] 测试模型调用...")

    # 可以修改这里的模型名称
    model_name = "qwen-turbo"  # 或 "qwen-plus", "qwen-max", "qwen-2190" 等
    test_message = "你好，请用一句话介绍你自己。"

    print(f"   模型: {model_name}")
    print(f"   测试消息: {test_message}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": test_message}],
            timeout=30,
            temperature=0.0
        )

        result = response.choices[0].message.content.strip()

        print("\n✅ 模型调用成功!")
        print(f"\n模型回复:\n{'-' * 60}")
        print(result)
        print('-' * 60)

        # 显示一些额外信息
        if hasattr(response, 'usage'):
            print(f"\nToken 使用情况:")
            print(f"  - 输入 tokens: {response.usage.prompt_tokens}")
            print(f"  - 输出 tokens: {response.usage.completion_tokens}")
            print(f"  - 总计 tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n❌ 模型调用失败: {e}")
        print(f"\n错误类型: {type(e).__name__}")
        return False

    finally:
        print("\n" + "=" * 60)

if __name__ == "__main__":
    success = test_qwen_connection()

    if success:
        print("\n🎉 测试通过！Qwen 模型连接正常。")
    else:
        print("\n❌ 测试失败，请检查配置。")
        print("\n常见问题排查:")
        print("1. 确认 .env 文件中的 API Key 是否正确")
        print("2. 确认 OPENAI_BASE_URL 是否设置正确")
        print("   通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("3. 确认模型名称是否正确（如 qwen-turbo, qwen-plus 等）")
        print("4. 检查网络连接是否正常")
