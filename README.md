# Legal Clause finer with LLMs

A smart document analysis system that leverages Large Language Models (LLMs) to efficiently search and analyze legal documents, reducing analysis time from hours to minutes while maintaining high recall for clause detection.

## Problem Statement

Legal professionals face several challenges when analyzing documents:
- Documents range from 10-300 pages with dense, domain-specific content
- Manual clause search is time-consuming and error-prone
- Non-standardized formats (scanned, hand-signed, digital)
- Need for high recall to avoid missing crucial clauses
- Semantic variations and synonyms make keyword search insufficient

## Setup

### Prerequisites
- Python 3.8+
- Azure OpenAI Service
- Azure Document Intelligence
- Azure AI Search

### Environment Variables
```bash
DOC_INT_ENDPOINT=your_document_intelligence_endpoint
DOC_INT_KEY=your_document_intelligence_key
ADA_ENDPOINT=your_ada_embedding_endpoint
ADA_KEY=your_ada_key
AI_SEARCH_ENDPOINT=your_search_endpoint
AI_SEARCH_KEY=your_search_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

### Installation with Poetry

1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Initialize Project
```bash
poetry init
```

3. Add Dependencies
```bash
poetry add fastapi uvicorn azure-ai-formrecognizer azure-search-documents openai numpy python-multipart azure-core tiktoken concurrent-futures
```

4. Create Virtual Environment and Install Dependencies
```bash
poetry install
```
5. Enter Poetry shell
```bash
poetry shell
```

6. Run the Application
```bash
uvicorn main:app --reload
```

## Project Structure
```
├──backend
    ├── pyproject.toml          # Poetry configuration
    ├── poetry.lock            # Lock file for dependencies
    ├── main.py               # FastAPI application
    ├── utils/
    │   ├──azure_utils.py       # Azure service clients
    │   ├── document_processor.py # Document processing logic
    │   └── dev/
    │       ├── azure_trial.py    # Experimental Azure integration code
    │       └── chunking_strategy_trial.py  # Chunking experiments
    ├── pipelines/
        ├── llm_pipeline.py     # LLM query processing

```

## Project Structure
```
├──backend
    ├── pyproject.toml          # Poetry configuration
    ├── poetry.lock            # Lock file for dependencies
    ├── main.py               # FastAPI application
    ├── utils/
    │   ├──azure_utils.py       # Azure service clients
    │   ├── document_processor.py # Document processing logic
    │   └── dev/
    │       ├── azure_trial.py    # Experimental Azure integration code
    │       └── chunking_strategy_trial.py  # Chunking experiments
    ├── pipelines/
        ├── llm_pipeline.py     # LLM query processing
```

### Experimental Code

**Azure Trial ipynb** (`utils/dev/azure_trial.py`)
- Blob storage operations and downloads
- Vector index creation in Azure AI Search
- Document chunking implementations
- Text embedding generation
- Index upload procedures
- Configuration testing

**Chunking Strategy Trial ipynb** (utils/dev/chunking_strategy_trial.py)
- Fixed-size chunking experiments
- Semantic-aware splitting
- Overlap configurations
- Performance benchmarking
- Quality assessment tools

The experimental code provides a sandbox for testing different approaches before implementing them in the production pipeline. This allows for:
- Testing various Azure service configurations
- Optimizing chunking strategies for legal documents
- Evaluating embedding quality
- Measuring search performance
- Validating integration points

All successful experiments are eventually migrated to the main production code in `utils/azure_utils.py` and `utils/document_processor.py`.

In the README.md, I'll update the /upload endpoint documentation to include all the available parameters. Here's the corrected version of that section:

## API Endpoints

### Upload Document
```http
POST /upload
```

**Request Parameters:**
- `file`: PDF document (Required)
- `title`: Document title (Optional)
- `link`: Document URL (Optional)
- `account`: Account name (Optional)
- `client_name`: Client name (Optional)
- `document_category`: Document category (Optional, defaults to "IMA")

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contract.pdf" \
  -F "title=Service Agreement 2024" \
  -F "link=https://example.com/doc" \
  -F "account=ACME Corp" \
  -F "client_name=John Doe" \
  -F "document_category=Contract"
```

**Response:**
```json
{
  "message": "Document processed successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_processed": 25
}
```

### Search Document
```http
POST /search
```

**Request:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "Find all clauses related to termination notice period",
  "conversation_history": []
}
```

**Response:**
```json
{
  "result": {
    "1. **Page: 12**
      - Under Section: 8.2 Termination
      - Section Summary: "Details the conditions and procedures for contract termination"
      - Cited Text: "Either party may terminate this Agreement by providing 90 days written notice..."

    "2. **Page: 15**
      - Under Section: 8.4 Material Breach
      - Section Summary: "Specifies the termination process in case of material breach"
      - Cited Text: "In case of material breach, notice period shall be reduced to 30 days..."
  }
}
```

## Features

- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent chunk processing
- **Token Management**: Implements tiktoken for accurate token counting
- **Error Handling**: Comprehensive error handling with custom LLMPipelineError
- **Recursive Summarization**: Handles large documents through recursive summary generation
- **Conversation History**: Supports context-aware queries with conversation history
- **Deterministic Output**: Uses low temperature settings for consistent responses

## Development

```bash
# Activate virtual environment
poetry shell

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --dev package-name

# Update dependencies
poetry update

# Run tests
poetry run pytest
```

## Limitations

- Maximum file size: 1 GB
- Supported document types: PDF only
- Maximum context window: 128k tokens
- Rate limits apply based on Azure service tier
- Maximum token limit per context chunk: 100,000



## Document Processing Assumptions

**Document Type & Processing**
- Documents are legal contracts processed through Azure Document Intelligence for text extraction
- Each document page becomes a single chunk, maintaining document structure and context

**Embedding & Storage**
- Page content is embedded using Azure OpenAI's ADA-002 model, generating 1500+ dimensional vectors
- Each chunk (page) is stored with comprehensive metadata including:
  - Page number
  - Document name
  - Document type
  - Document link
  - Account information
  - Client details
  - Content vector
  - Title vector

**Search Infrastructure**
- Azure AI Search is used for vector storage and retrieval
- Chunks are indexed with both content and metadata for efficient searching
- Search supports semantic configuration for improved results

**Retrieval Process**
- Documents are retrieved using pagination with a batch size of 100
- Maximum retrieval limit of 10,000 results per search
- Uses semantic configuration "my-semantic-config" for enhanced search capabilities
