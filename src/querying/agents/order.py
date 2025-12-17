"""Order agent using OpenAI function calling."""
import asyncio
from openai import AsyncOpenAI
from src.querying.tools.retrieval import get_product_search_function
from src.querying.tools.order import (
    get_add_to_cart_function,
    get_edit_item_in_cart_function,
    get_remove_from_cart_function,
    get_view_cart_function,
    get_shipping_info_function,
    get_create_shipping_info_function,
    get_edit_shipping_info_function,
    get_get_orders_function,
    get_purchase_function
)
from src.utils.llm import create_chat_completion_with_timeout, run_db_operation_with_timeout


class OrderAgent:
    """
    Agent specialized in handling order-related queries.
    Can search for products, add to cart, view cart, and complete purchases.
    """
    
    def __init__(self, client: AsyncOpenAI, min_similarity: float = 0.75, vectorstore=None):
        """
        Initialize the order agent.
        
        Args:
            client: OpenAI client instance
            min_similarity: Minimum similarity threshold for retrieval
            vectorstore: Optional pre-initialized products vectorstore
        """
        self.client = client
        self.model = "gpt-4o-mini"  # Can be made configurable
        self.min_similarity = min_similarity
        self.vectorstore = vectorstore
        
        # Define tools
        self.tools = [
            get_product_search_function(min_similarity),
            get_add_to_cart_function(),
            get_edit_item_in_cart_function(),
            get_remove_from_cart_function(),
            get_view_cart_function(),
            get_shipping_info_function(),
            get_create_shipping_info_function(),
            get_edit_shipping_info_function(),
            get_get_orders_function(),
            get_purchase_function()
        ]
        
        self.system_prompt = (
            "You are Shoplytic's Order Agent.\n\n"
            "Your job is to decide the SINGLE NEXT ACTION needed to move the user toward completing their request.\n\n"
            "RULES:\n"
            "- In each turn, you may either:\n"
            "  (A) Call exactly ONE tool, OR\n"
            "  (B) Ask the user for missing required information\n"
            "- NEVER call more than one tool in a single response\n"
            "- NEVER perform actions without tools\n"
            "- NEVER assume state — use tools to check state\n\n"
            "- NEVER caluclate cart total and items in it by yourself, always use view_cart"
            "Shopping flow:\n"
            "1. Search products (search_products)\n"
            "2. Add to cart (add_to_cart)\n"
            "3. View cart (view_cart)\n"
            "4. Collect shipping info (get_shipping_info, create_shipping_info, edit_shipping_info)\n"
            "5. Purchase (purchase)\n\n"
            "If information is missing (shipping info, voucher code, product choice), ask the user clearly what is needed.\n\n"
            "Tool usage:\n"
            "- search_products: Find products. Extract filters: price (below/under/cheap → max_price, above/premium → min_price), category (laptops/phones/watche(s)/smartwatch → Electronics, shoes/clothes → Clothing, headphones → Accessories), brand, featured status\n"
            "- add_to_cart: Add product to cart (requires product_id and optional quantity)\n"
            "- view_cart: Check current cart contents\n"
            "- edit_item_in_cart: Update quantity of item in cart\n"
            "- remove_from_cart: Remove item from cart\n"
            "- get_shipping_info: Check if shipping information exists\n"
            "- create_shipping_info: Create shipping information (requires fullName, address, city, zipCode)\n"
            "- edit_shipping_info: Update shipping information\n"
            "- get_orders: Get order information (optional order_id, otherwise returns 5 most recent)\n"
            "- purchase: Complete purchase (requires voucher_code)"
        )
    
    async def _execute_tool(self, tool_call, session_id: str, query: str) -> tuple[str, list]:
        """
        Execute a single tool call and return result and sources.
        
        Args:
            tool_call: Tool call object from LLM
            session_id: User session identifier
            query: Original user query
            
        Returns:
            Tuple of (tool_result, sources_list)
        """
        import json
        from langchain_core.documents import Document
        from src.querying.tools.retrieval import execute_product_search
        from src.querying.tools.order import (
            execute_add_to_cart,
            execute_edit_item_in_cart,
            execute_remove_from_cart,
            execute_view_cart,
            execute_get_shipping_info,
            execute_create_shipping_info,
            execute_edit_shipping_info,
            execute_get_orders,
            execute_purchase
        )
        from src.utils.cart import cart_manager
        from data.database.connection import SessionLocal
        from data.database.order_models import ShippingInfo
        
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        sources = []
        
        # Hard-gate purchase execution
        if function_name == "purchase":
            # Check cart has items
            cart = cart_manager.get_cart(session_id)
            if not cart or len(cart) == 0:
                return ("Error: Your cart is empty. Please add items to your cart before purchasing.", sources)
            
            # Check shipping info exists
            db = SessionLocal()
            try:
                shipping_info = db.query(ShippingInfo).filter(
                    ShippingInfo.session_id == session_id
                ).first()
                if not shipping_info:
                    return ("Error: Please provide shipping information before purchasing. Use create_shipping_info or provide your shipping details.", sources)
            finally:
                db.close()
            
            # Check voucher code provided
            if not function_args.get("voucher_code"):
                return ("Error: Please provide a voucher code to complete your purchase.", sources)
        
        # Execute tool
        if function_name == "search_products":
            search_query = function_args.get("query", query)
            try:
                tool_result, docs_with_similarity = await run_db_operation_with_timeout(
                    execute_product_search,
                    timeout=15.0,
                    timeout_error_message="Error: Product search timed out. Please try again.",
                    query=search_query,
                    k=function_args.get("k", 3),
                    category=function_args.get("category"),
                    brand=function_args.get("brand"),
                    min_price=function_args.get("min_price"),
                    max_price=function_args.get("max_price"),
                    is_featured=function_args.get("is_featured"),
                    min_similarity=self.min_similarity,
                    vectorstore=self.vectorstore
                )
                sources.extend(docs_with_similarity)
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "add_to_cart":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_add_to_cart,
                    timeout_error_message="Error: Adding to cart timed out. Please try again.",
                    session_id=session_id,
                    product_id=function_args.get("product_id"),
                    quantity=function_args.get("quantity", 1)
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "edit_item_in_cart":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_edit_item_in_cart,
                    timeout_error_message="Error: Updating cart item timed out. Please try again.",
                    session_id=session_id,
                    product_id=function_args.get("product_id"),
                    quantity=function_args.get("quantity")
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "remove_from_cart":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_remove_from_cart,
                    timeout_error_message="Error: Removing item from cart timed out. Please try again.",
                    session_id=session_id,
                    product_id=function_args.get("product_id")
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "view_cart":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_view_cart,
                    timeout_error_message="Error: Viewing cart timed out. Please try again.",
                    session_id=session_id
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "get_shipping_info":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_get_shipping_info,
                    timeout_error_message="Error: Retrieving shipping information timed out. Please try again.",
                    session_id=session_id
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "create_shipping_info":
            try:
                shipping_data = function_args.get("shipping_data", {})
                tool_result = await run_db_operation_with_timeout(
                    execute_create_shipping_info,
                    timeout_error_message="Error: Saving shipping information timed out. Please try again.",
                    session_id=session_id,
                    shipping_data=shipping_data
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "edit_shipping_info":
            try:
                shipping_data = function_args.get("shipping_data", {})
                tool_result = await run_db_operation_with_timeout(
                    execute_edit_shipping_info,
                    timeout_error_message="Error: Updating shipping information timed out. Please try again.",
                    session_id=session_id,
                    shipping_data=shipping_data
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "get_orders":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_get_orders,
                    timeout=10.0,
                    timeout_error_message="Error: Retrieving orders timed out. Please try again.",
                    session_id=session_id,
                    order_id=function_args.get("order_id")
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        elif function_name == "purchase":
            try:
                tool_result = await run_db_operation_with_timeout(
                    execute_purchase,
                    timeout=15.0,
                    timeout_error_message="Error: Processing purchase timed out. Please try again.",
                    session_id=session_id,
                    voucher_code=function_args.get("voucher_code")
                )
            except asyncio.TimeoutError as e:
                tool_result = str(e)
        else:
            tool_result = f"Error: Unknown function '{function_name}'"
        
        return (tool_result, sources)
    
    async def invoke(self, query: str, session_id: str, conversation_history: list = None) -> tuple[str, list, dict]:
        """
        Process a query using the order agent with loop-based execution.
        
        Args:
            query: User's question or query
            session_id: User session identifier for cart management
            conversation_history: Previous conversation messages (includes product_ids from previous searches)
            
        Returns:
            Tuple of (agent response text, list of source documents, query parameters dict)
        """
        from langchain_core.documents import Document
        import json
        
        # Build messages with conversation history if provided
        messages = [{"role": "system", "content": self.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        
        sources = []
        query_params = {}  # Track query parameters used in search_products
        
        # Loop until completion (max 6 steps for safety)
        for step in range(6):
            try:
                response = await create_chat_completion_with_timeout(
                    client=self.client,
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
            except asyncio.TimeoutError:
                return ("I apologize, but the request took too long to process. Please try again.", sources, query_params)
            
            message = response.choices[0].message
            
            # Log tool calls
            if message.tool_calls:
                print(f"[ORDER AGENT] Step {step+1}: Tool calls returned: {len(message.tool_calls)}")
                for i, tc in enumerate(message.tool_calls):
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    print(f"  Tool call #{i+1}: {tc.function.name} with args: {args}, tool_call_id: {tc.id}")
            else:
                print(f"[ORDER AGENT] Step {step+1}: No tool calls returned, content: {message.content[:100] if message.content else 'None'}")
            
            # 1️⃣ If no tool call → we're done
            if not message.tool_calls:
                return (message.content or "", sources, query_params)

            # 2️⃣ Enforce: no duplicate tool calls with identical signature (function + args)
            signatures = set()
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                key = (tc.function.name, json.dumps(args, sort_keys=True))
                if key in signatures:
                    raise RuntimeError(
                        f"OrderAgent violated rule: duplicate tool calls with same args: {tc.function.name}"
                    )
                signatures.add(key)

            # Assistant message with ALL tool calls
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

            # 3️⃣ Execute each tool call
            for tool_call in message.tool_calls:
                # Capture query params for search_products
                if tool_call.function.name == "search_products":
                    function_args = json.loads(tool_call.function.arguments)
                    search_query = function_args.get("query", query)
                    query_params["query"] = search_query
                    if function_args.get("category"):
                        query_params["category"] = function_args.get("category")
                    if function_args.get("brand"):
                        query_params["brand"] = function_args.get("brand")
                    if function_args.get("min_price") is not None:
                        query_params["min_price"] = function_args.get("min_price")
                    if function_args.get("max_price") is not None:
                        query_params["max_price"] = function_args.get("max_price")
                    if function_args.get("is_featured") is not None:
                        query_params["is_featured"] = function_args.get("is_featured")

                tool_result, tool_sources = await self._execute_tool(tool_call, session_id, query)
                sources.extend(tool_sources)

                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
        
        # If we've exhausted steps, return the last response
        return ("I apologize, but the request took too many steps to complete. Please try again.", sources, query_params)
