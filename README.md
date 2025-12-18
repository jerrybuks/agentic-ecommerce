# Agentic Ecommerce Platform

An ecommerce platform with AI agents and admin product management.

## Live Demo & Monitoring

- **Product Demo**: [https://agentic-ecommerce-fe.vercel.app/](https://agentic-ecommerce-fe.vercel.app/)
- **API Documentation**: [https://agentic-ecommerce.onrender.com/docs](https://agentic-ecommerce.onrender.com/docs)
- **Status Page**: [https://stats.uptimerobot.com/5z2EBCHShQ](https://stats.uptimerobot.com/5z2EBCHShQ)
- **Report**: [https://github.com/jerrybuks/agentic-ecommerce/blob/main/reports/REPORT.MD](https://github.com/jerrybuks/agentic-ecommerce/blob/main/reports/REPORT.MD)

## Setup

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Configure environment:**
   - Create a `.env` file with the following variables:
     ```
     DATABASE_URL=postgresql://user:password@host:port/database # You can use [Neon](https://neon.com/) its free
     OPENAI_API_KEY=your_openai_api_key_here
     OPENAI_API_BASE=https://openrouter.ai/api/v1  # Optional: for OpenRouter or other providers
     LANGFUSE_PUBLIC_KEY=langfuse_public_key
     LANGFUSE_SECRET_KEY=langfuse_secret_key
     LANGFUSE_BASE_URL=https://cloud.langfuse.com
     ```
   - Update `DATABASE_URL` with your Neon PostgreSQL connection string
   - Add your OpenAI API key for embeddings
   - If using OpenRouter, set `OPENAI_API_BASE` to `https://openrouter.ai/api/v1`
   - add your langfuse project details

3. **Run the application:**
   ```bash
   uvicorn src.main:app --reload
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Endpoints

### User Endpoints

#### Query & Chat
- `POST /user/query` - Process a user query through the multi-agent system (chatbot)

#### Vouchers
- `POST /user/vouchers/generate` - Generate a new $2000 USD voucher code (returns existing unused voucher if available)

#### Cart
- `GET /user/cart` - Get the current user's shopping cart

#### Orders
- `GET /user/orders` - Get all orders for the current user (includes shipping address)

#### Products
- `GET /user/products` - Get products with search and filters (pagination, category, brand, price range, tags)
- `GET /user/products/featured` - Get featured products
- `GET /user/products/{product_id}` - Get product details by ID

### Admin API Endpoints

#### Products

- `POST /admin/products/` - Create a new product
- `GET /admin/products/` - Get all products (with optional filters)
- `GET /admin/products/{product_id}` - Get product by ID
- `GET /admin/products/sku/{sku}` - Get product by SKU
- `PUT /admin/products/{product_id}` - Update a product
- `PATCH /admin/products/{product_id}` - Partially update a product

## Project Structure

```
agentic-ecommerce/
├── data/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py      # Database connection and session management
│   │   ├── product_model.py   # Product SQLAlchemy model
│   │   └── product_schema.py  # Pydantic schemas for validation
│   ├── handbooks/
│   │   └── general_handbook.md  # Customer handbook for indexing
│   ├── jsonl/                 # Generated chunk files (gitignored)
│   │   ├── product_chunks.jsonl
│   │   └── handbook_chunks.jsonl
│   └── vector_store/          # ChromaDB vector store (gitignored)
│                               # Contains: products and general_handbook collections
└── src/
    ├── main.py                # FastAPI application entry point
    ├── config.py              # Configuration settings
    ├── build_index.py         # Indexing pipeline orchestration
    ├── indexing/
    │   ├── __init__.py
    │   ├── parsing.py         # Product and handbook parsing
    │   ├── chunking.py        # Text chunking strategies
    │   └── embeddings.py      # Embedding generation and ChromaDB storage
    ├── routes/
    │   ├── __init__.py
    │   └── admin.py           # Admin API routes
    └── utils/
        ├── __init__.py
        └── storage.py          # Storage utilities (ChromaDB operations)
```

## Testing

Run golden dataset tests against the chatbot API:

```bash
# Run all tests
python -m tests.test_runner

# Run by category
python -m tests.test_runner --category cart_operations

# Run single test
python -m tests.test_runner --id product_search_001

# Save results to file
python -m tests.test_runner --save
```

Test cases are defined in `data/golden_data/test_cases.json`.

## Database

The application uses PostgreSQL 17 with SQLAlchemy ORM. The database tables are automatically created when the application starts.

### Product Model

The Product model includes:
- Basic info (name, SKU, descriptions)
- Pricing (price, cost_price)
- Inventory (stock_quantity, low_stock_threshold)
- Product details (weight, dimensions, category, tags)
- Media (images, primary_image)
- Status (is_active, is_featured)
- Brand information
- Timestamps (created_at, updated_at)

## Indexing Pipeline

The platform includes a semantic search indexing pipeline using LangChain and ChromaDB.

### Building the Index

To build the product index for semantic search:

```bash
python src/build_index.py
```

**Note:** The index is always fully rebuilt (existing index is cleared) when you run this command.

### Options

- `--batch-size`: Number of products to process per batch (default: 100)
- `--chunk-size`: Maximum chunk size in characters (default: 1000)
- `--chunk-overlap`: Chunk overlap in characters (default: 200)
- `--include-inactive`: Include inactive products in index

### Indexing Components

- **Parsing** (`src/indexing/parsing.py`): Loads and parses products from the database and markdown handbooks
- **Chunking** (`src/indexing/chunking.py`): Intelligently chunks product documents and markdown handbooks
- **Embeddings** (`src/indexing/embeddings.py`): Generates OpenAI embeddings and stores in ChromaDB
- **Storage** (`src/utils/storage.py`): Utilities for storing documents in ChromaDB

The index uses:
- **Embedding Model**: OpenAI `text-embedding-ada-002`
- **Vector Store**: ChromaDB with cosine similarity
- **Storage**: `data/vector_store/` directory
- **Collections**: 
  - `products` - Product documents
  - `general_handbook` - Customer handbook documents

