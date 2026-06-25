import sys
from types import ModuleType
# Workaround for newer langchain-community versions where vertexai is moved/removed
mock_vertex = ModuleType('langchain_community.chat_models.vertexai')
mock_vertex.ChatVertexAI = None
sys.modules['langchain_community.chat_models.vertexai'] = mock_vertex

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ingestion import load_and_chunk
from vectorstore import create_vectorstore, get_retriever
from generator import get_rag_chain

def run_evaluation():
    print("Setting up evaluation environment...")
    
    # 1. Load data and setup RAG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, "..", "data", "employee_handbook.md"))
    if not os.path.exists(data_path):
        print(f"Test data not found at {data_path}")
        return

    chunks = load_and_chunk(data_path)
    for chunk in chunks:
        chunk.metadata["source"] = "employee_handbook.md"
        
    vectorstore = create_vectorstore(chunks)
    retriever = get_retriever(vectorstore)
    rag_chain = get_rag_chain(retriever)

    # 2. Define test set
    questions = [
        "How many days of sick leave do employees get?",
        "What is the remote work policy?",
        "Can I fix my broken laptop myself?"
    ]
    
    ground_truths = [
        "Employees receive 8 days of paid sick leave annually.",
        "Employees can work remotely up to three days a week. Mondays and Thursdays are mandatory in-office days for local employees.",
        "No, employees should not attempt to fix hardware issues themselves. They must submit a ticket or email IT support."
    ]

    print("Generating answers for evaluation dataset...")
    answers = []
    contexts = []
    
    for q in questions:
        # Get context
        docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in docs])
        
        # Get answer
        ans = rag_chain.invoke(q)
        answers.append(ans)

    # 3. Format for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data)

    # 4. Evaluate
    print("Running ragas evaluation (this requires an OpenAI API key by default)...")
    try:
        # Note: Ragas uses OpenAI by default for evaluation models.
        # To run this fully locally, you'd need to configure Ragas to use local Ollama models.
        result = evaluate(
            dataset = dataset, 
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
        )
        print("\n=== Evaluation Results ===")
        print(result)
    except Exception as e:
        print(f"\nEvaluation failed. Note: Ragas typically requires an OPENAI_API_KEY environment variable. Error: {e}")

if __name__ == "__main__":
    run_evaluation()
