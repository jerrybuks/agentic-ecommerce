"""Utility functions for LLM operations."""
import asyncio
from typing import Any, Callable, Dict, Optional
from langfuse.openai import AsyncOpenAI
from src.config import settings

# Database operation timeout (5 seconds - should be much faster)
DB_TIMEOUT = 5.0


async def create_chat_completion_with_timeout(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    timeout: Optional[float] = None,
    default_error_message: str = "I apologize, but the request took too long to process. Please try again.",
    **kwargs
) -> Any:
    """
    Create a chat completion with timeout handling.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name to use
        messages: List of messages for the chat completion
        timeout: Timeout in seconds (defaults to settings.llm_timeout)
        default_error_message: Error message to return on timeout
        **kwargs: Additional arguments to pass to chat.completions.create
        
    Returns:
        Chat completion response, or raises TimeoutError if timeout occurs
        
    Raises:
        asyncio.TimeoutError: If the request exceeds the timeout
    """
    if timeout is None:
        timeout = settings.llm_timeout
    
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        ),
        timeout=timeout
    )


async def run_db_operation_with_timeout(
    func: Callable,
    timeout: float = DB_TIMEOUT,
    timeout_error_message: str = "Error: Database operation timed out. Please try again.",
    *args,
    **kwargs
) -> Any:
    """
    Run a blocking database operation in a thread pool with timeout.
    
    Args:
        func: The blocking function to execute
        timeout: Timeout in seconds (defaults to DB_TIMEOUT)
        timeout_error_message: Error message to return on timeout
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result from func
        
    Raises:
        asyncio.TimeoutError: If the operation exceeds the timeout (with error message)
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # Create a new TimeoutError with the custom message
        error = asyncio.TimeoutError()
        error.args = (timeout_error_message,)
        raise error

