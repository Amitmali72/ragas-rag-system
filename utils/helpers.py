"""General-purpose utility functions for the Knowledge Copilot pipeline.

Includes a timing decorator, human-readable time formatting, content
hashing for deduplication, and chat export to Markdown.
"""

import functools
import hashlib
import time
from typing import Any, Callable, Dict, List, Tuple


def timer_decorator(func: Callable) -> Callable:
    """Decorator that measures a function's execution time.

    The decorated function returns a tuple of (original_result, time_ms)
    instead of just the original result.

    Usage:
        @timer_decorator
        def my_func(x):
            return x * 2

        result, elapsed_ms = my_func(5)

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that returns (result, elapsed_ms).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start: float = time.time()
        result = func(*args, **kwargs)
        elapsed_ms: float = (time.time() - start) * 1000.0
        return result, round(elapsed_ms, 2)

    return wrapper


def format_time(ms: float) -> str:
    """Convert milliseconds to a human-readable string.

    - Values ≥ 1000 ms are shown as seconds (e.g., "1.23s").
    - Values < 1000 ms are shown as milliseconds (e.g., "456ms").

    Args:
        ms: Time in milliseconds.

    Returns:
        Formatted time string.
    """
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f}s"
    return f"{ms:.0f}ms"


def hash_content(text: str) -> str:
    """Return an MD5 hex digest of a text string.

    Useful for deduplication of chunks by content.

    Args:
        text: The text to hash.

    Returns:
        32-character hexadecimal hash string.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def export_chat_to_markdown(messages: List[Dict[str, str]]) -> str:
    """Convert a list of chat messages to a downloadable Markdown string.

    Each message dict should have 'role' (e.g., "user", "assistant") and
    'content' keys.

    Args:
        messages: List of chat message dicts.

    Returns:
        A Markdown-formatted string representing the conversation.
    """
    lines: List[str] = ["# Knowledge Copilot — Chat Export", ""]

    for msg in messages:
        role: str = msg.get("role", "unknown").capitalize()
        content: str = msg.get("content", "")

        if role.lower() == "user":
            lines.append(f"### 🧑 User")
        elif role.lower() == "assistant":
            lines.append(f"### 🤖 Assistant")
        else:
            lines.append(f"### {role}")

        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
