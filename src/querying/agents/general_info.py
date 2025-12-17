"""General information agent using OpenAI function calling."""
import asyncio
from langfuse.openai import AsyncOpenAI
from langfuse import get_client
from src.querying.tools.retrieval import get_handbook_retrieval_function
from src.utils.llm import create_chat_completion_with_timeout


class GeneralInfoAgent:
    """
    Agent specialized in answering general information questions.
    Uses the handbook vector store to retrieve company policies, product offerings,
    refund policies, shipping information, etc.
    """
    
    def __init__(self, client: AsyncOpenAI, min_similarity: float = 0.75, vectorstore=None):
        """
        Initialize the general info agent.
        
        Args:
            client: OpenAI client instance
            min_similarity: Minimum similarity threshold for retrieval
            vectorstore: Optional pre-initialized handbook vectorstore
        """
        self.client = client
        self.model = "gpt-4o-mini"  # Can be made configurable
        self.min_similarity = min_similarity
        self.vectorstore = vectorstore
        
        # Define tools
        self.tools = [
            get_handbook_retrieval_function(min_similarity)
        ]
        
        self.system_prompt = (
            "You are Shoplytic's customer service agent. Answer questions about Shoplytic's company policies, "
            "FAQs, shipping/returns, and general information using the provided handbook content. "
            "You represent Shoplytic, not yourself. Be helpful and accurate."
        )
    
    async def invoke(self, query: str) -> tuple[str, list]:
        """
        Process a query using the general info agent.
        
        Args:
            query: User's question or query
            
        Returns:
            Tuple of (agent response text, list of source documents)
        """
        from src.querying.tools.retrieval import execute_handbook_retrieval
        
        langfuse = get_client()
        
        with langfuse.start_as_current_observation(
            as_type="span",
            name="general-info-agent",
            input={"query": query}
        ) as agent_span:
            # Directly call retrieval (no initial LLM call needed since we only have one tool)
            tool_result, docs_with_similarity = execute_handbook_retrieval(
                query=query,
                k=3,
                min_similarity=self.min_similarity,
                vectorstore=self.vectorstore
            )
            
            sources = list(docs_with_similarity)
            
            # Generate final response with retrieved context
            # Include retrieved information in the user message
            user_message = f"{query}\n\nRelevant information from the handbook:\n{tool_result}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            try:
                final_response = await create_chat_completion_with_timeout(
                    client=self.client,
                    model=self.model,
                    messages=messages
                )
                response_text = final_response.choices[0].message.content
                agent_span.update(output={"response": response_text[:500] if response_text else "", "sources_count": len(sources)})
                return (response_text, sources)
            except asyncio.TimeoutError:
                agent_span.update(output={"error": "timeout"})
                return ("I apologize, but the request took too long to process. Please try again.", sources)
