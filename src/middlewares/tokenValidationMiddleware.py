"""Token validation middleware for query endpoints."""
import json
import tiktoken
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

# Token limit for queries
MAX_QUERY_TOKENS = 300

# Initialize tiktoken encoder (using cl100k_base for GPT-4/GPT-3.5)
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


class TokenValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate query token count before processing."""
    
    async def dispatch(self, request: StarletteRequest, call_next):
        # Only validate POST requests to /user/query
        if request.method == "POST" and request.url.path == "/user/query":
            try:
                # Read the body
                body_bytes = await request.body()
                
                if body_bytes:
                    data = json.loads(body_bytes)
                    query = data.get("query", "")
                    
                    # Count tokens
                    token_count = len(tiktoken_encoder.encode(query))
                    
                    if token_count > MAX_QUERY_TOKENS:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "detail": f"Query exceeds maximum token limit. Got {token_count} tokens, maximum allowed is {MAX_QUERY_TOKENS}."
                            }
                        )
                    
                    # Create a new request with the body for downstream handlers
                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    
                    request._receive = receive
                    
            except json.JSONDecodeError:
                pass  # Let the endpoint handle invalid JSON
            except Exception as e:
                print(f"[TOKEN_VALIDATION] Error: {e}")
        
        return await call_next(request)


