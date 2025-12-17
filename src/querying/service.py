"""Query service that manages the orchestrator agent with memory."""
from typing import Optional
from src.querying.agents.orchestrator import OrchestratorAgent
from src.utils.memory import ConversationMemory
from src.config import settings


class QueryService:
    """
    Service for handling user queries through the multi-agent system.
    Manages the orchestrator agent and provides memory/context sharing.
    """
    
    def __init__(self, handbook_vectorstore=None, products_vectorstore=None):
        """
        Initialize the query service with orchestrator and memory.
        
        Args:
            handbook_vectorstore: Optional pre-initialized handbook vectorstore
            products_vectorstore: Optional pre-initialized products vectorstore
        """
        # Initialize memory
        self.memory = ConversationMemory(max_queries=10)
        
        # Initialize orchestrator with vector stores
        self.orchestrator = OrchestratorAgent(
            self.memory,
            handbook_vectorstore=handbook_vectorstore,
            products_vectorstore=products_vectorstore
        )
    
    async def query(
        self,
        user_query: str,
        session_id: str,
        min_similarity: Optional[float] = None
    ) -> dict:
        """
        Process a user query through the multi-agent system.
        
        Args:
            user_query: The user's question or request
            session_id: Session identifier for maintaining conversation context
            min_similarity: Optional minimum similarity score threshold
                           (default: from config)
            
        Returns:
            Dictionary with response, routing_mode, and agents_used
        """
        min_sim = min_similarity if min_similarity is not None else settings.default_similarity_threshold
        
        result = await self.orchestrator.invoke(
            query=user_query,
            session_id=session_id,
            min_similarity=min_sim
        )
        
        return result
