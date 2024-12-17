from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
import os
import uuid
from typing import List
import numpy as np
from pipelines import LLMQueryPipeline
from pydantic import BaseModel
from utils.document_processor import *
app = FastAPI()

class SearchRequest(BaseModel):
    document_id: str
    query: str
    conversation_history: List[dict] = []

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    title: str = None,
    link: str = None,
    account: str = None,
    client_name: str = None,
    document_category: str = "IMA"  # default value
):
    try:
        # Create a temporary file to store the upload
        temp_path = f"/tmp/{str(uuid.uuid4())}{os.path.splitext(file.filename)[1]}"
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Generate a unique document ID
        doc_id = str(uuid.uuid4()).upper()
        
        # Process the document with additional metadata
        documents = process_document(
            local_path=temp_path, 
            doc_id=doc_id, 
            title=title or file.filename,
            link=link or "",
            account=account or "N/A",
            client_name=client_name or "N/A",
            document_category=document_category
        )
        
        # Upload to Azure AI Search
        upload_to_vector_index(documents)
        
        # Cleanup temporary file
        os.remove(temp_path)
        
        return JSONResponse(
            content={
                "message": "Document processed successfully",
                "document_id": doc_id,
                "chunks_processed": len(documents)
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_document(request: SearchRequest):
    try:
        pipeline = LLMQueryPipeline()
        result = pipeline.process_query(
            query=request.query,
            document_id=request.document_id,
            conversation_history=request.conversation_history
        )
        
        return JSONResponse(
            content={
                "result": result
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
