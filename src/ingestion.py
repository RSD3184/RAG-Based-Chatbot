import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(filepath: str):
    """Loads a document based on its file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(filepath)
    elif ext == '.md':
        loader = UnstructuredMarkdownLoader(filepath)
    elif ext == '.txt':
        loader = TextLoader(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return loader.load()

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """Splits documents into smaller chunks for vectorization."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def load_and_chunk(filepath: str):
    """Convenience function to load and chunk a file."""
    docs = load_document(filepath)
    chunks = chunk_documents(docs)
    return chunks
