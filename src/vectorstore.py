import os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

DB_DIR = "./chroma_db"

@lru_cache(maxsize=1)
def get_embedding_model():
    # Using a fast and effective sentence transformer
    # all-MiniLM-L6-v2 is small and fast, standard for simple RAG
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(chunks):
    """Creates a Chroma vector database from document chunks."""
    embedding_model = get_embedding_model()
    
    # Store locally so we don't have to re-embed on every run
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    return vectorstore

def get_vectorstore():
    """Loads the existing Chroma vector database."""
    if not os.path.exists(DB_DIR):
        return None
        
    embedding_model = get_embedding_model()
        
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )
    return vectorstore

def get_retriever(vectorstore, k=4, filter=None):
    """Returns a retriever interface from the vectorstore."""
    # We can configure search type (e.g., 'mmr' for diversity, or 'similarity')
    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
