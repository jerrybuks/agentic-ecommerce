"""Custom orchestrator agent with OpenAI function calling."""
import asyncio
from typing import List, Dict, Any, Optional, Literal
from langfuse.openai import AsyncOpenAI
from langfuse import get_client
from src.config import settings
from src.utils.memory import ConversationMemory
from src.utils.llm import create_chat_completion_with_timeout
from src.utils.evaluation import evaluate_response_async


class OrchestratorAgent:
    """
    Custom orchestrator that routes queries to sub-agents.
    Supports single, sequential, and parallel agent execution.
    """
    
    ROUTING_MODES = Literal["single", "sequential", "parallel"]
    
    def __init__(self, memory: ConversationMemory, handbook_vectorstore=None, products_vectorstore=None):
        """
        Initialize the orchestrator.
        
        Args:
            memory: Conversation memory manager
            handbook_vectorstore: Optional pre-initialized handbook vectorstore
            products_vectorstore: Optional pre-initialized products vectorstore
        """
        self.memory = memory
        self.handbook_vectorstore = handbook_vectorstore
        self.products_vectorstore = products_vectorstore
        
        # Initialize OpenAI client (async)
        client_kwargs = {
            "api_key": settings.openai_api_key
        }
        if settings.openai_api_base:
            client_kwargs["base_url"] = settings.openai_api_base
        self.client = AsyncOpenAI(**client_kwargs)
        
        self.model = settings.chat_model
        
        # Define orchestrator functions (routing to sub-agents)
        self.functions = [
            {
                "type": "function",
                "function": {
                    "name": "query_general_info",
                    "description": "Query the general information agent for company policies, product offerings, refund policies, shipping information, and general company information. Use for: policy questions, FAQ, company info, shipping/return policies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's question about general information"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_order_agent",
                    "description": "Query the order agent for product search, order creation, order status, and product recommendations. Use for: product search, purchasing, order management, product details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's question or request related to orders"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        self.system_prompt = (
            "You are Shoplytic's orchestrator. Route queries to sub-agents. "
            "ALWAYS call a routing function (except greetings).\n\n"
            "ROUTING:\n"
            "- query_general_info: Policies, FAQs, shipping/returns, company info\n"
            "- query_order_agent: Products, orders, cart, purchasing, shipping info, vouchers\n\n"
            "CRITICAL: All product-related queries → query_order_agent. "
            "Never answer product questions directly.\n\n"
            "Use parallel for independent questions, sequential for dependent ones."
        )
    
    async def _call_sub_agent(self, agent_name: str, query: str, min_similarity: float, session_id: str, conversation_history: list = None) -> tuple[str, list, dict]:
        """
        Call a sub-agent to process a query.
        
        Args:
            agent_name: Name of the sub-agent ('general_info' or 'order')
            query: User query
            min_similarity: Minimum similarity threshold for retrieval
            session_id: User session identifier (for order agent cart management)
            conversation_history: Previous conversation messages (includes product_ids from previous searches)
            
        Returns:
            Tuple of (sub-agent response, list of source documents, query parameters dict)
        """
        from src.querying.agents.general_info import GeneralInfoAgent
        from src.querying.agents.order import OrderAgent
        
        if agent_name == "general_info":
            agent = GeneralInfoAgent(self.client, min_similarity, self.handbook_vectorstore)
            response, sources = await agent.invoke(query)
            return (response, sources, {})  # General info doesn't have product search query params
        elif agent_name == "order":
            agent = OrderAgent(self.client, min_similarity, self.products_vectorstore)
            response, sources, query_params = await agent.invoke(query, session_id, conversation_history)
            return (response, sources, query_params)
        else:
            return (f"Unknown agent: {agent_name}", [], {})
    
    async def invoke(
        self,
        query: str,
        session_id: str,
        min_similarity: float = 0.75
    ) -> Dict[str, Any]:
        """
        Process a user query through the orchestrator.
        
        Args:
            query: User's question or request
            session_id: Session identifier for memory
            min_similarity: Minimum similarity threshold for retrieval
            
        Returns:
            Dictionary with response, routing_mode, and agents_used
        """
        langfuse = get_client()
        
        # Wrap entire query lifecycle in a trace
        with langfuse.start_as_current_observation(
            as_type="span",
            name="query-lifecycle",
            input={"query": query, "session_id": session_id}
        ) as root_trace:
            # Get conversation history
            history = self.memory.get_messages(session_id)
            current_message = {"role": "user", "content": query}
            messages = history + [current_message]
            
            # Call orchestrator with functions (this combines order routing and tool selection)
            # The LLM will determine order routing mode implicitly by how it calls the functions
            try:
                response = await create_chat_completion_with_timeout(
                    client=self.client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *messages
                    ],
                    tools=self.functions,
                    tool_choice="auto",  # Model decides, but system prompt enforces routing
                    max_tokens=settings.llm_max_tokens_orchestrator
                )
            except asyncio.TimeoutError:
                response_text = "I apologize, but the request took too long to process. Please try again."
                self.memory.add_query(session_id, query, response_text, [])
                root_trace.update(output={"response": response_text, "error": "timeout"})
                return {
                    "response": response_text,
                    "routing_mode": "direct",
                    "agents_used": [],
                    "sources": [],
                    "query_params": {}
                }
            
            message = response.choices[0].message
            
            # Log tool calls from orchestrator
            if message.tool_calls:
                import json
                print(f"[ORCHESTRATOR] Tool calls returned: {len(message.tool_calls)}")
                for i, tc in enumerate(message.tool_calls):
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    print(f"  Tool call #{i+1}: {tc.function.name} with args: {args}, tool_call_id: {tc.id}")
            else:
                print(f"[ORCHESTRATOR] No tool calls returned, content: {message.content[:100] if message.content else 'None'}")
            
            agents_used = []
            sub_agent_responses = []
            all_sources = []
            query_params = {}  # Collect query parameters from sub-agents
            routing_mode = "single"  # Default, will be determined from tool calls
            
            # Handle tool calls
            if message.tool_calls:
                import json
                tool_messages = []

                # If multiple order-agent calls are returned, collapse into one using the original query
                order_tool_calls = [tc for tc in message.tool_calls if tc.function.name == "query_order_agent"]
                if len(order_tool_calls) > 1:
                    first_order = order_tool_calls[0]
                    first_order.function.arguments = json.dumps({"query": query})
                    # keep non-order calls plus the first order call
                    filtered = [tc for tc in message.tool_calls if tc.function.name != "query_order_agent"]
                    filtered.insert(0, first_order)
                    message.tool_calls = filtered
                    print(f"[ORCHESTRATOR] Collapsed {len(order_tool_calls)} order calls into 1 with original query")
                
                # Prepare agent calls
                agent_calls = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Map function name to agent
                    if function_name == "query_general_info":
                        agent_name = "general_info"
                        agents_used.append("general_info")
                    elif function_name == "query_order_agent":
                        agent_name = "order"
                        agents_used.append("order")
                    else:
                        continue
                    
                    sub_query = function_args.get("query", query)
                    agent_calls.append({
                        "tool_call": tool_call,
                        "agent_name": agent_name,
                        "query": sub_query
                    })
                
                # Determine routing mode from tool calls
                # If multiple different agents called in one response, they can run in parallel
                # If same agent called multiple times, it's sequential
                unique_agents = len(set(agents_used))
                if len(agents_used) > 1 and unique_agents > 1:
                    routing_mode = "parallel"
                elif len(agents_used) > 1:
                    routing_mode = "sequential"
                else:
                    routing_mode = "single"
                
                # Execute agents based on routing mode
                # Pass conversation history so agents can see previous search results with product_ids
                if routing_mode == "parallel":
                    # Execute all agents in parallel
                    tasks = [
                        self._call_sub_agent(call["agent_name"], call["query"], min_similarity, session_id, messages)
                        for call in agent_calls
                    ]
                    results = await asyncio.gather(*tasks)
                    
                    # Process results
                    for call, (sub_response, sub_sources, sub_query_params) in zip(agent_calls, results):
                        sub_agent_responses.append({
                            "agent": call["agent_name"],
                            "response": sub_response
                        })
                        all_sources.extend(sub_sources)
                        # Merge query params (order agent will have product search params)
                        query_params.update(sub_query_params)
                        tool_messages.append({
                            "role": "tool",
                            "content": sub_response,
                            "tool_call_id": call["tool_call"].id
                        })
                else:
                    # Execute agents sequentially
                    for call in agent_calls:
                        sub_response, sub_sources, sub_query_params = await self._call_sub_agent(
                            call["agent_name"], call["query"], min_similarity, session_id, messages
                        )
                        sub_agent_responses.append({
                            "agent": call["agent_name"],
                            "response": sub_response
                        })
                        all_sources.extend(sub_sources)
                        # Merge query params (order agent will have product search params)
                        query_params.update(sub_query_params)
                        tool_messages.append({
                            "role": "tool",
                            "content": sub_response,
                            "tool_call_id": call["tool_call"].id
                        })
                
                # Add assistant message with tool_calls to messages for LLM synthesis
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Add all tool messages
                messages.extend(tool_messages)
                
                # For single agent flows, response is already synthesized - use it directly
                if routing_mode == "single" and len(agent_calls) == 1:
                    response_text = sub_agent_responses[0]["response"]
                else:
                    # Multiple agents or sequential calls - need synthesis
                    final_response = await create_chat_completion_with_timeout(
                        client=self.client,
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are summarizing tool results for the user. Do NOT call any tools."},
                            *messages
                        ],
                        max_tokens=settings.llm_max_tokens_agent
                    )
                    final_message = final_response.choices[0].message
                    response_text = final_message.content or ""
            else:
                # No tool calls - orchestrator responded directly (e.g., for greetings)
                print(f"[ORCHESTRATOR] Direct response (no routing): {message.content[:100] if message.content else 'None'}")
                response_text = message.content or ""
                routing_mode = "direct"
                # No agents used, no sources
            
            # Store in memory with only product sources (for product_id retrieval by order agent)
            # Filter to only include sources with product_id (from order agent), exclude handbook sources
            product_sources = []
            for source in all_sources:
                if isinstance(source, tuple):
                    doc, similarity = source
                else:
                    doc = source
                
                from langchain_core.documents import Document
                if isinstance(doc, Document) and doc.metadata.get("product_id"):
                    # Only store product sources (not handbook sources)
                    product_sources.append(source)
            
            self.memory.add_query(session_id, query, response_text, product_sources)
            
            # Update root trace with final output
            root_trace.update(output={
                "response": response_text[:500],  # Truncate for brevity
                "routing_mode": routing_mode,
                "agents_used": list(set(agents_used)) if agents_used else []
            })
            
            # === LLM-AS-A-JUDGE QUALITY EVALUATION ===
            # Fire off async evaluation (non-blocking)
            trace_id = langfuse.get_current_trace_id()
            if trace_id:
                asyncio.create_task(
                    evaluate_response_async(
                        query=query,
                        response=response_text,
                        trace_id=trace_id,
                        agents_used=list(set(agents_used)) if agents_used else [],
                        session_id=session_id
                    )
                )
            
            return {
                "response": response_text,
                "routing_mode": routing_mode,
                "agents_used": list(set(agents_used)) if agents_used else [],
                "sources": all_sources,
                "query_params": query_params
            }
