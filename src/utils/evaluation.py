"""
LLM-as-a-Judge Evaluation Service for Quality Scoring.

Evaluates chatbot responses across multiple quality dimensions using an LLM judge.
Scores are submitted to Langfuse for tracking and analytics.
"""
import json
import asyncio
from typing import Optional
from langfuse import get_client
from langfuse.openai import AsyncOpenAI
from src.config import settings


# Quality dimensions to evaluate (each scored 1-10)
QUALITY_DIMENSIONS = [
    "relevance",      # How relevant is the response to the user's query?
    "accuracy",       # Is the information provided factually correct?
    "completeness",   # Does the response fully address the user's request?
    "clarity",        # Is the response clear and easy to understand?
    "helpfulness"     # How helpful is the response in achieving the user's goal?
]


EVALUATION_PROMPT = """You are an expert evaluator for an e-commerce chatbot called Shoplytic.
Your task is to evaluate the quality of the chatbot's response to a user query.

**User Query:**
{query}

**Chatbot Response:**
{response}

**Agents Used:**
{agents_used}

Evaluate the response on the following dimensions, scoring each from 1 to 10:

1. **Relevance** (1-10): How relevant is the response to the user's query?
   - 1-3: Completely off-topic or irrelevant
   - 4-6: Partially relevant but misses key aspects
   - 7-9: Mostly relevant with minor gaps
   - 10: Perfectly relevant and on-point

2. **Accuracy** (1-10): Is the information provided factually correct?
   - 1-3: Contains significant errors or misinformation
   - 4-6: Some inaccuracies or uncertain claims
   - 7-9: Mostly accurate with minor issues
   - 10: Completely accurate

3. **Completeness** (1-10): Does the response fully address the user's request?
   - 1-3: Missing critical information or actions
   - 4-6: Addresses some parts but incomplete
   - 7-9: Mostly complete with minor omissions
   - 10: Fully addresses all aspects of the request

4. **Clarity** (1-10): Is the response clear and easy to understand?
   - 1-3: Confusing, poorly structured, or hard to follow
   - 4-6: Somewhat clear but could be improved
   - 7-9: Clear and well-organized
   - 10: Exceptionally clear and well-presented

5. **Helpfulness** (1-10): How helpful is the response in achieving the user's goal?
   - 1-3: Not helpful, may frustrate the user
   - 4-6: Somewhat helpful but user needs to do more work
   - 7-9: Very helpful, guides user effectively
   - 10: Extremely helpful, exceeds expectations

Respond with a JSON object in this exact format:
{{
    "overall_quality": <1-10>,
    "relevance": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
    "accuracy": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
    "completeness": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
    "clarity": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
    "helpfulness": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
    "overall_reasoning": "<brief overall assessment>"
}}

Return ONLY the JSON object, no other text."""


async def evaluate_response(
    query: str,
    response: str,
    trace_id: str,
    agents_used: list[str],
    session_id: Optional[str] = None
) -> Optional[dict]:
    """
    Evaluate a chatbot response using LLM-as-a-Judge.
    
    Args:
        query: The user's original query
        response: The chatbot's response
        trace_id: Langfuse trace ID to attach scores to
        agents_used: List of agents that handled the query
        session_id: Optional session ID for context
        
    Returns:
        Dictionary with evaluation scores, or None if evaluation failed
    """
    langfuse = get_client()
    
    # Initialize OpenAI client for evaluation
    client_kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_api_base:
        client_kwargs["base_url"] = settings.openai_api_base
    client = AsyncOpenAI(**client_kwargs)
    
    # Format the evaluation prompt
    eval_prompt = EVALUATION_PROMPT.format(
        query=query,
        response=response[:2000],  # Truncate long responses
        agents_used=", ".join(agents_used) if agents_used else "none"
    )
    
    try:
        # Call the LLM judge
        eval_response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use a cost-effective model for evaluation
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON only."},
                {"role": "user", "content": eval_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p
        )
        
        # Parse the evaluation result
        eval_content = eval_response.choices[0].message.content
        eval_result = json.loads(eval_content)
        
        # Submit scores to Langfuse
        # 1. Overall quality score
        langfuse.create_score(
            trace_id=trace_id,
            name="overall_quality",
            value=float(eval_result.get("overall_quality", 5)),
            data_type="NUMERIC",
            comment=eval_result.get("overall_reasoning", "")
        )
        
        # 2. Individual dimension scores
        for dimension in QUALITY_DIMENSIONS:
            dim_data = eval_result.get(dimension, {})
            if isinstance(dim_data, dict):
                score = dim_data.get("score", 5)
                reasoning = dim_data.get("reasoning", "")
            else:
                score = dim_data if isinstance(dim_data, (int, float)) else 5
                reasoning = ""
            
            langfuse.create_score(
                trace_id=trace_id,
                name=f"quality_{dimension}",
                value=float(score),
                data_type="NUMERIC",
                comment=reasoning
            )
        
        # Flush to ensure scores are sent
        langfuse.flush()
        
        return eval_result
        
    except json.JSONDecodeError as e:
        print(f"[EVALUATION] Failed to parse evaluation JSON: {e}")
        return None
    except Exception as e:
        print(f"[EVALUATION] Evaluation failed: {e}")
        return None


async def evaluate_response_async(
    query: str,
    response: str,
    trace_id: str,
    agents_used: list[str],
    session_id: Optional[str] = None
):
    """
    Fire-and-forget async evaluation.
    Runs evaluation in background without blocking the main response.
    
    Args:
        query: The user's original query
        response: The chatbot's response  
        trace_id: Langfuse trace ID to attach scores to
        agents_used: List of agents that handled the query
        session_id: Optional session ID for context
    """
    try:
        await evaluate_response(
            query=query,
            response=response,
            trace_id=trace_id,
            agents_used=agents_used,
            session_id=session_id
        )
    except Exception as e:
        # Don't let evaluation errors affect the main flow
        print(f"[EVALUATION] Background evaluation error: {e}")
