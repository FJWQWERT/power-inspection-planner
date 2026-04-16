"""
硅基流动 (SiliconFlow) LLM API 客户端封装
- 基于 OpenAI 兼容接口
- 模型: deepseek-ai/DeepSeek-V3
- 支持错误重试和超时设置
"""

import os
import sys
import io
import time
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

# 强制 stdout 使用 UTF-8 编码（仅在未设置时执行，避免重复包装）
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 配置
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3"
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
TIMEOUT = 60     # 秒


def _get_client():
    """获取 OpenAI 兼容客户端"""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 SILICONFLOW_API_KEY 环境变量。"
            "请通过 set SILICONFLOW_API_KEY=your_key (Windows) "
            "或 export SILICONFLOW_API_KEY=your_key (Linux/Mac) 设置。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT,
    )


def call_llm(prompt, system_message=None, temperature=0.7, max_tokens=2000):
    """
    调用硅基流动 LLM API

    参数:
        prompt:         用户输入的提示词
        system_message: 系统消息（可选）
        temperature:    生成温度，0~1，越高越随机
        max_tokens:     最大生成 token 数

    返回:
        str: 模型生成的回复文本

    异常:
        在重试耗尽后抛出最后一次遇到的异常
    """
    client = _get_client()

    # 构建消息列表
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            print(f"  [重试 {attempt}/{MAX_RETRIES}] 连接/超时错误: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except RateLimitError as e:
            last_error = e
            print(f"  [重试 {attempt}/{MAX_RETRIES}] 频率限制: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # 递增等待

        except APIError as e:
            last_error = e
            print(f"  [重试 {attempt}/{MAX_RETRIES}] API错误: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise last_error


# ============ 测试代码 ============
if __name__ == "__main__":
    sep = "=" * 60
    print(sep)
    print("  硅基流动 LLM API 调用测试")
    print(sep)

    print(f"\n  Base URL: {BASE_URL}")
    print(f"  Model:    {MODEL}")
    print(f"  Timeout:  {TIMEOUT}s")
    print(f"  Retries:  {MAX_RETRIES}")

    print(f"\n{'─' * 50}")
    print("  测试: 调用 call_llm('你好，请介绍一下你自己')")
    print(f"{'─' * 50}\n")

    try:
        result = call_llm("你好，请介绍一下你自己")
        print(f"  模型回复:\n")
        for line in result.strip().split("\n"):
            print(f"    {line}")
        print(f"\n{sep}")
        print("  测试通过!")
        print(sep)
    except Exception as e:
        print(f"\n  测试失败: {e}")
        print(sep)
        sys.exit(1)
