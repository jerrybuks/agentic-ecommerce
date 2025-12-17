"""Memory management for conversation context."""
from typing import List, Dict, Any
from collections import deque


class ConversationMemory:
    """Manages conversation memory for a session (last 10 queries)."""
    
    def __init__(self, max_queries: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_queries: Maximum number of queries to keep in memory
        """
        self.max_queries = max_queries
        self.sessions: Dict[str, deque] = {}
    
    def add_query(self, session_id: str, query: str, response: str, sources: List[Any] = None):
        """
        Add a query-response pair to memory.
        
        Args:
            session_id: Session identifier
            query: User query
            response: Agent response
            sources: Optional list of source documents (for product search results, etc.)
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_queries)
        
        self.sessions[session_id].append({
            "query": query,
            "response": response,
            "sources": sources or []
        })
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of query-response pairs
        """
        return list(self.sessions.get(session_id, deque()))
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get messages in OpenAI format for a session, including product information from sources.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in OpenAI format, with product information included in assistant responses
        """
        history = self.get_history(session_id)
        messages = []
        for item in history:
            messages.append({"role": "user", "content": item["query"]})
            
            # Include assistant response, and append product information from sources if available
            response_content = item["response"]
            sources = item.get("sources", [])
            
            # If there are sources with product information, append it to the response
            # This makes product_ids available for future queries
            if sources:
                from langchain_core.documents import Document
                product_info_parts = []
                for source in sources:
                    if isinstance(source, tuple):
                        doc, similarity = source
                    else:
                        doc = source
                    
                    if isinstance(doc, Document):
                        product_id = doc.metadata.get("product_id")
                        if product_id:
                            product_info_parts.append(
                                f"Product ID: {product_id}\n"
                                f"Brand: {doc.metadata.get('brand', 'N/A')}\n"
                                f"Category: {doc.metadata.get('category', 'N/A')}\n"
                                f"Price: ${doc.metadata.get('price', 'N/A')}"
                            )
                
                if product_info_parts:
                    # Append product information to the response so it's visible in conversation context
                    response_content += "\n\n[Previous search results with product_ids:]\n" + "\n\n---\n\n".join(product_info_parts)
            
            messages.append({"role": "assistant", "content": response_content})
        
        return messages

