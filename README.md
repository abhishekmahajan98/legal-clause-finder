# Legal Document Analysis with LLMs

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

5. Run the Application
```bash
poetry run uvicorn main:app --reload
```

## Project Structure
```
├── pyproject.toml          # Poetry configuration
├── poetry.lock            # Lock file for dependencies
├── main.py               # FastAPI application
├── document_processor.py # Document processing logic
├── azure_utils.py       # Azure service clients
└── llm_pipeline.py     # LLM query processing
```

## API Endpoints

### Upload Document
```http
POST /upload
```

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contract.pdf" \
  -F "title=Service Agreement 2024" \
  -F "link=https://example.com/doc"
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

- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent chunk processing[1]
- **Token Management**: Implements tiktoken for accurate token counting[1]
- **Error Handling**: Comprehensive error handling with custom LLMPipelineError[1]
- **Recursive Summarization**: Handles large documents through recursive summary generation[1]
- **Conversation History**: Supports context-aware queries with conversation history[1]
- **Deterministic Output**: Uses low temperature settings for consistent responses[1]

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

- Maximum file size: 50MB
- Supported languages: English only
- Maximum context window: 32k tokens
- Rate limits apply based on Azure service tier
- Maximum token limit per context chunk: 100,000[1]

## Future Improvements

- Multi-language support
- Custom domain-specific models
- Improved chunking strategies
- Real-time processing for smaller documents
- Batch processing capabilities

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/18058410/0d717939-e1e7-40ff-b1ef-3af606cf0dd9/paste.txt