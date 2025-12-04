"""
Token 预测服务
用于在前后端调用时估算文本分块与合并的 token 消耗。
"""
from pathlib import Path
from typing import Dict, Any

from tokenizer import count_tokens
from splitter import get_splitter


def estimate_tokens(file_path: str) -> Dict[str, Any]:
    """
    估算 token 消耗。
    Returns:
        {
            "total_tokens": int,
            "chunk_tokens": int,
            "chunk_responses": int,
            "merge_tokens": int,
            "total_estimated": int,
            "chunk_count": int,
        }
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    splitter = get_splitter()

    total_tokens = count_tokens(text)
    chunks = splitter.split_text(text)
    chunk_tokens = sum(count_tokens(chunk) for chunk in chunks)
    chunk_responses = int(chunk_tokens * 0.3)  # 估算响应
    merge_tokens = int(total_tokens * 0.1)    # 估算合并消耗

    return {
        "total_tokens": total_tokens,
        "chunk_tokens": chunk_tokens,
        "chunk_responses": chunk_responses,
        "merge_tokens": merge_tokens,
        "total_estimated": chunk_tokens + chunk_responses + merge_tokens,
        "chunk_count": len(chunks),
    }
