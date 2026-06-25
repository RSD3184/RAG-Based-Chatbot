import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs_with_sources(docs):
    """Formats retrieved documents to include source metadata for the LLM."""
    formatted_chunks = []
    for i, doc in enumerate(docs):
        # Extract metadata (like source file or page, defaulting to 'Unknown Source')
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "")
        source_str = f"Source: {source}" + (f", Page: {page}" if page else "")
        formatted_chunks.append(f"[{i+1}] {source_str}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)

import requests

def get_ollama_model():
    """Dynamically detects installed Ollama models, falling back to Llama3."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            # Prefer llama3 if available
            for name in model_names:
                if "llama3" in name:
                    return name
            # Fallback to the first available model
            if model_names:
                return model_names[0]
    except Exception:
        pass
    return "llama3"

def get_rag_chain(retriever):
    """Creates a RAG chain with a strict prompt against hallucination."""
    
    # Dynamically select the model based on what is locally available
    model_name = get_ollama_model()
    llm = Ollama(model=model_name)
    
    system_prompt = (
        "You are an AI assistant for answering questions based on the provided documents. "
        "Your task is to answer the user's question using ONLY the provided context.\n\n"
        "RULES:\n"
        "1. If the answer is not contained in the provided context, you MUST say exactly: "
        "\"I don't know the answer based on the provided documents.\"\n"
        "2. Do not hallucinate or use outside knowledge.\n"
        "3. When you provide an answer, you MUST cite the source of the information by referencing "
        "the document and page number provided in the context (e.g., [Source: handbook.pdf, Page: 2]).\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # Build the LangChain LCEL (LangChain Expression Language) pipeline
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
