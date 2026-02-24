import openai
from openai import OpenAI

from typing import Dict, List, Optional, Any, Union
import os

from dotenv import load_dotenv
import time as _time
import logging

_logger = logging.getLogger(__name__)

load_dotenv()

MAX_RETRIES = 5
BACKOFF_BASE = 2        # seconds; actual wait = base * 2^attempt (exponential)
BACKOFF_MAX = 60         # cap


model2api = {
    "DeepSeek-R1": os.getenv("DEEPSEEK_R1_API_KEY"),
    "o1-mini-2024-09-12": os.getenv("O1_MINI_API_KEY"),
    "o3-2025-04-16": os.getenv("O3_API_KEY"),
    "gpt-5-2025-08-07": os.getenv("GPT_5_API_KEY"),
    "gpt-4o-2024-05-13": os.getenv("GPT_4O_API_KEY"),
    "gpt-4o-mini-2024-07-18": os.getenv("GPT_4O_MINI_API_KEY"),
    "gpt-5.2-2025-12-11": os.getenv("GPT_5_2_API_KEY"),
    "gpt-5.2-2025-12-11-300": os.getenv("GPT_5_2_API_KEY_300"),
    "gpt-oss-120b": os.getenv("GPT_OSS_120B_API_KEY"),
    "gemini-2.5-pro-preview-05-06": os.getenv("GEMINI_2_5_PRO_PREVIEW_05_06_API_KEY"),
    "openai_qwen3-14b": os.getenv("QWEN3_API_KEY"),
    "openai_qwen3-32b": os.getenv("QWEN3_API_KEY")
}
# for parallel control
model_name_mapping = {
    "gpt-5.2-2025-12-11-300": "gpt-5.2-2025-12-11"
}

def close_proxy():
    os.environ["no_proxy"]=""
    os.environ["http_proxy"]=""
    os.environ["https_proxy"]=""

def open_proxy():
    os.environ["no_proxy"]=""
    os.environ["http_proxy"]="http://sys-proxy-rd-relay.byted.org:8118"
    os.environ["https_proxy"]="http://sys-proxy-rd-relay.byted.org:8118"

def query_llm(
    model_name: str,
    **kwargs: Any
):
    """
    Query LLM with automatic retry and exponential backoff on failure.
    Retries on: rate limit, timeout, connection, server errors.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if model_name in ["qwen3-8b","qwen3-14b","gpt-4o-2024-08-06","DeepSeek-R1","o1-mini-2024-09-12"]:
                name = model_name
                if name == "DeepSeek-R1":
                    name = "deepseek-r1"
                if name == "o1-mini-2024-09-12":
                    name = "o1-mini"
                max_tokens = kwargs.get("max_tokens",8192)
                if "max_tokens" in kwargs:
                    del kwargs["max_tokens"]
                return query_llm_outer(model_name=name, max_tokens=max_tokens,**kwargs)
            else:
                return query_llm_inhouse(model_name=model_name, **kwargs)

        except (openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError) as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
            _logger.warning(
                f"[error info] {str(e)}"
                f"[query_llm] {model_name} {type(e).__name__} (attempt {attempt+1}/{MAX_RETRIES+1}), "
                f"retrying in {wait}s..."
            )
            print(f"  ⚠ LLM error: {type(e).__name__}, backoff {wait}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES+1})", flush=True)
            _time.sleep(wait)

        except Exception as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
            _logger.warning(
                f"[error info] {str(e)}"
                f"[query_llm] {model_name} Unexpected {type(e).__name__}: {e} "
                f"(attempt {attempt+1}/{MAX_RETRIES+1}), retrying in {wait}s..."
            )
            print(f"  ⚠ LLM error: {type(e).__name__}: {str(e)[:80]}, backoff {wait}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES+1})", flush=True)
            _time.sleep(wait)

    raise last_exc

def query_llm_outer(
    model_name: str,
    messages: Union[List[Dict[str, str]], str],
    api_key: str=os.getenv("YUNWU_API_KEY"),
    base_url: str = "https://yunwu.ai/v1/",
    system: Optional[str] = None,
    max_tokens: Optional[int] = 8192,
    temperature: float = 1.0,
    top_p: float = 0.7,
    **kwargs: Any
) -> Dict[str, Optional[str]]:
    """
    查询 LLM API 并返回推理内容和响应
    
    Args:
        model_name: 模型名称
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]，或者直接传入字符串（会自动包装为 user role）
        api_key: API 密钥
        base_url: API 基础 URL，默认为 "https://yunwu.ai/v1/"
        system: 可选的 system 消息（字符串）
        max_tokens: 最大生成 token 数，默认为 None（由模型决定）
        temperature: 温度参数，默认为 1.0
        top_p: top_p 参数，默认为 0.7
        **kwargs: 其他传递给 API 的参数
    
    Returns:
        包含 'reasoning_content' 和 'response' 的字典
    """
    open_proxy()
    # 处理 messages：如果是字符串，自动包装
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # 处理 system：如果提供了 system 参数，添加到 messages 开头
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 准备 API 调用参数
    api_params = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        # "top_p": top_p,
        **kwargs
    }
    # 如果提供了 max_tokens，添加到参数中
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    # if "max_tokens" in api_params:
    #     del api_params["max_tokens"]
    chat_completion = client.chat.completions.create(**api_params)
    message = chat_completion.choices[0].message
    
    # 提取 reasoning_content（如果存在）
    reasoning_content = None
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        reasoning_content = message.reasoning_content
    elif hasattr(message, 'reasoning') and message.reasoning:
        reasoning_content = message.reasoning
    
    # 提取 response
    response = message.content if message.content else ""
    close_proxy()
    return {
        'reasoning_content': reasoning_content,
        'response': response
    }

def query_llm_inhouse(
    model_name: str,
    messages: Union[List[Dict[str, str]], str],
    system: Optional[str] = None,
    max_tokens: Optional[int] = 8192,
    temperature: float = 1.0,
    top_p: float = 0.7,
    **kwargs: Any
) -> Dict[str, Optional[str]]:
    global model2api
    global model_name_mapping
    
    close_proxy()
    # 处理 messages：如果是字符串，自动包装
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # 处理 system：如果提供了 system 参数，添加到 messages 开头
    if system is not None:
        messages = [{"role": "system", "content": system}] + messages


    client = openai.AzureOpenAI(
        api_key=model2api[model_name],
        azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl",
        api_version="2024-03-01-preview",
    )
    extra_args = {
        "top_p": top_p,
        "temperature": temperature,
    }
    if model_name in ["DeepSeek-R1"]:
        response = client.chat.completions.create(
            model=model_name_mapping.get(model_name,model_name),
            messages=messages,
            max_tokens=max_tokens,
            extra_headers={
                "X-TT-LOGID": "${your_logid}"
            },
            **extra_args
        )
    # elif "qwen" in model_name:
    #     client = openai.AzureOpenAI(
    #         azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
    #         api_version="2024-03-01-preview",
    #         api_key=model2api[model_name]
    #     )
    #     extra_args["enable_thinking"] = False
    #     response = client.chat.completions.create(
    #         model=model_name_mapping.get(model_name,model_name),
    #         messages=messages,
    #         max_tokens=max_tokens,
    #         stream=True,
    #         extra_headers={
    #             "X-TT-LOGID": "${your_logid}"
    #         },
    #         **extra_args
    #     )
    else:
        response = client.chat.completions.create(
            model=model_name_mapping.get(model_name,model_name),
            messages=messages,
            max_tokens=max_tokens,
            extra_headers={
                "X-TT-LOGID": "${your_logid}"
            },
        )
    model_response = response.choices[0].message.content
    reasoning_content =   response.choices[0].message.reasoning_content
    open_proxy()
    return {
        'reasoning_content': reasoning_content,
        'response': model_response
    }

if __name__ == "__main__":
    ans = query_llm(
        model_name="DeepSeek-R1",
        messages="你好?请问人生的意义是什么",
    )
    print(ans['response'])
    print(ans['reasoning_content'])
