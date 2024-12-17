from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
import numpy as np
from azure_utils import get_ai_search_client
def get_document_intelligence_client():
    endpoint = str(os.getenv("DOC_INT_ENDPOINT"))
    key = str(os.getenv("DOC_INT_KEY"))
    return DocumentAnalysisClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )

def extract_document_structure(local_path):
    document_analysis_client = get_document_intelligence_client()
    with open(local_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", document=f
        )
    return poller.result()

def get_embedding_model_client():
    ada_endpoint = str(os.getenv("ADA_ENDPOINT"))
    ada_key = str(os.getenv("ADA_KEY"))
    return AzureOpenAI(
        azure_endpoint=ada_endpoint,
        api_key=ada_key,
        api_version="2023-05-15"
    )

def create_embeddings(text, max_tokens=8191):
    ada_deployment = "text-embedding-ada-002"
    chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    openai_client = get_embedding_model_client()
    embeddings = []
    
    for chunk in chunks:
        response = openai_client.embeddings.create(
            input=chunk,
            model=ada_deployment
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    
    if len(embeddings) > 1:
        return np.mean(embeddings, axis=0).tolist()
    else:
        return embeddings[0]

def extract_text_from_page(page):
    text = ""
    for line in page.lines:
        text += line.content + "\n"
    return text.strip()

def process_document(local_path, doc_id, title, link):
    document_result = extract_document_structure(local_path)
    
    documents = []
    for page_num, page in enumerate(document_result.pages, 1):
        page_text = extract_text_from_page(page)
        chunk_id = f"{doc_id}-p{page_num}"
        chunk_title = f"{title} - Page {page_num}"
        
        documents.append({
            "id": chunk_id,
            "document_id": doc_id,
            "title": chunk_title,
            "content": page_text,
            "page_number": page_num,
            "document_category": "IMA",
            "document_title": title,
            "link": link,
            "titleVector": create_embeddings(chunk_title),
            "contentVector": create_embeddings(page_text)
        })
    
    return documents

def upload_to_vector_index(documents):
    search_client = get_ai_search_client()
    result = search_client.upload_documents(documents)
    print(f"Uploaded {len(result)} documents")
