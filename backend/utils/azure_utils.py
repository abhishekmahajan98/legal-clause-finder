from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
import os

def get_ai_search_client():
    """
    Initialize and return an Azure AI Search client.
    """
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX")
    
    if not all([search_endpoint, search_key, index_name]):
        raise ValueError("Missing required Azure Search environment variables")
    
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential
    )
    return search_client

def get_gpt4o_client():
    """
    Initialize and return an Azure OpenAI client and deployment name.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if not all([azure_endpoint, azure_key, deployment_name]):
        raise ValueError("Missing required Azure OpenAI environment variables")
    
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version="2023-05-15"
    )
    
    return client, deployment_name
