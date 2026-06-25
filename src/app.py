import os
import streamlit as st
from ingestion import load_document, chunk_documents
from vectorstore import create_vectorstore, get_retriever, get_vectorstore
from generator import get_rag_chain

# --- Page Config ---
st.set_page_config(page_title="RAG QA Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
/* Base theme */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Custom Header Gradient */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #9B5DE5, #3F8EFC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

/* Chat message bubbles styling */
[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Assistant icon color tweak */
[data-testid="stChatMessageAvatarUser"] {
    background-color: #9B5DE5;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #3F8EFC;
}

/* Button styling to match the theme */
div.stButton > button {
    background: linear-gradient(45deg, #9B5DE5, #3F8EFC);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(45deg, #3F8EFC, #9B5DE5);
    color: white;
    box-shadow: 0 4px 15px rgba(63, 142, 252, 0.4);
    transform: translateY(-1px);
}

</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

if "processed_docs" not in st.session_state:
    if st.session_state.vectorstore:
        try:
            # Dynamically fetch unique source documents from the database metadata
            db_data = st.session_state.vectorstore.get()
            metadatas = db_data.get("metadatas", [])
            unique_docs = list(set(m.get("source_doc") for m in metadatas if m and "source_doc" in m))
            st.session_state.processed_docs = unique_docs
        except Exception:
            st.session_state.processed_docs = []
    else:
        st.session_state.processed_docs = []

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {doc: [] for doc in st.session_state.processed_docs}

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Document Library")
    
    st.markdown("Upload documents to build your knowledge base.")
    uploaded_file = st.file_uploader("Upload a document", type=['txt', 'md', 'pdf'])
    
    if uploaded_file is not None:
        doc_name = uploaded_file.name
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save temp file
                temp_path = f"temp_{doc_name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    st.toast("Loading document...")
                    docs = load_document(temp_path)
                    
                    st.toast("Chunking text...")
                    chunks = chunk_documents(docs)
                    
                    # Add metadata to chunks so we can filter by doc_name
                    for chunk in chunks:
                        chunk.metadata["source_doc"] = doc_name
                    
                    st.toast("Creating embeddings and storing in VectorDB...")
                    # If vectorstore exists, we add to it, else create new
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = create_vectorstore(chunks)
                    else:
                        st.session_state.vectorstore.add_documents(chunks)
                    
                    if doc_name not in st.session_state.processed_docs:
                        st.session_state.processed_docs.append(doc_name)
                    
                    if doc_name not in st.session_state.chat_histories:
                        st.session_state.chat_histories[doc_name] = []
                        
                    st.success(f"Successfully processed {doc_name}!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.divider()
    st.subheader("Current Knowledge Base")
    if not st.session_state.processed_docs:
        st.info("No documents uploaded yet.")
        active_doc = None
    else:
        active_doc = st.radio("Select Active Document for Chat:", st.session_state.processed_docs)
        
        # Delete document button
        st.divider()
        st.subheader("🗑️ Manage Documents")
        doc_to_delete = st.selectbox("Select document to remove:", st.session_state.processed_docs, key="delete_select")
        if st.button("Delete Document", type="secondary"):
            try:
                # Remove embeddings from ChromaDB using metadata filter
                collection = st.session_state.vectorstore._collection
                collection.delete(where={"source_doc": doc_to_delete})
                
                # Remove from processed docs list
                st.session_state.processed_docs.remove(doc_to_delete)
                
                # Remove chat history
                if doc_to_delete in st.session_state.chat_histories:
                    del st.session_state.chat_histories[doc_to_delete]
                
                st.success(f"Deleted '{doc_to_delete}' from the knowledge base.")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting document: {e}")

# --- Main Chat Area ---
st.title("🤖 Intelligent QA Chatbot")

if not active_doc:
    st.markdown("""
    Welcome to the RAG QA System. To get started:
    1. Upload a document from the sidebar.
    2. Click **Process Document**.
    3. Select the document from the Knowledge Base and start chatting!
    """)
else:
    st.subheader(f"Chatting with: `{active_doc}`")
    
    # Initialize chat history if not already initialized
    if active_doc not in st.session_state.chat_histories:
        st.session_state.chat_histories[active_doc] = []

    # Display chat history for the active document
    for message in st.session_state.chat_histories[active_doc]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("Show Citations", expanded=False):
                    for i, doc in enumerate(message["citations"]):
                        st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")

    # Chat Input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Append user message to history
        st.session_state.chat_histories[active_doc].append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant thinking...
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get retriever with filter for current doc
                    retriever = get_retriever(st.session_state.vectorstore, filter={"source_doc": active_doc})
                    
                    # Get source documents first
                    source_docs = retriever.invoke(prompt)
                    
                    # Generate the answer using the chain
                    rag_chain = get_rag_chain(retriever)
                    answer = rag_chain.invoke(prompt)
                    
                    # Render answer
                    st.markdown(answer)
                    
                    # Render citations
                    if source_docs:
                        with st.expander("Show Citations", expanded=False):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")
                    
                    # Append assistant message to history
                    st.session_state.chat_histories[active_doc].append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": source_docs
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
