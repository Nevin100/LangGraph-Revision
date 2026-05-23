# RAG (Retrieval Augmented Generation) - instead of LLM answering from its training data, it first retrieves relevant documents from your own knowledge base, then generates an answer based on those docs.

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

# State 
# Custom state — query, retrieved docs, and final answer
class RAGState(TypedDict):
    query: str
    documents: List[Document]
    answer: str


# LLM + Embeddings 
llm = ChatGroq(model="llama-3.3-70b-versatile")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample documents — in a real app, these would come from your knowledge base (like files, databases, etc.)
docs = [
    Document(page_content="LangGraph is a framework for building stateful multi-agent applications using graphs.", metadata={"source": "langgraph_docs"}),
    Document(page_content="Nodes in LangGraph are Python functions that take state as input and return updated state.", metadata={"source": "langgraph_docs"}),
    Document(page_content="Edges in LangGraph define the flow between nodes. Conditional edges allow dynamic routing.", metadata={"source": "langgraph_docs"}),
    Document(page_content="SqliteSaver is a checkpointer that persists graph state to a SQLite database.", metadata={"source": "langgraph_docs"}),
    Document(page_content="ToolNode is a built-in LangGraph node that automatically executes tool calls made by the LLM.", metadata={"source": "langgraph_docs"}),
    Document(page_content="Human in the Loop allows pausing graph execution for human review using interrupt() and Command(resume=...).", metadata={"source": "langgraph_docs"}),
]

# Chroma vectorstore — it embeds the documents and allows us to retrieve relevant ones based on a query. In a real app, you would set up your vectorstore with your actual knowledge base documents.
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # retrieve top - 2 

# Nodes Functions
def retrieve(state: RAGState):
    """Node 1 — Retrieve relevant docs based on the query"""
    print(f"\n[Retrieve] Query: {state['query']}")

    documents = retriever.invoke(state["query"])

    print(f"[Retrieve] {len(documents)} docs retrieved:")
    for i, doc in enumerate(documents):
        print(f"  Doc {i+1}: {doc.page_content[:60]}...")

    return {"documents": documents}

def generate(state: RAGState):
    """Node 2 — retrieved docs + query → LLM → answer"""
    print("\n[Generate] LLM generating answer based on retrieved docs...")

    context = "\n\n".join([doc.page_content for doc in state["documents"]])

    messages = [
        SystemMessage(content=f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have information about this."

Context:
{context}"""),
        HumanMessage(content=state["query"])
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}

# Graph 
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# Query function 
def ask(question: str):
    result = graph.invoke({"query": question})
    print(f"Q: {question}")
    print(f"A: {result['answer']}")

# Main 
if __name__ == "__main__":
    print(" Test 1: LangGraph question ")
    ask("What is LangGraph?")

    print("\n Test 2: Nodes question ")
    ask("What are nodes in LangGraph?")

    print("\n Test 3: HITL question ")
    ask("How does Human in the Loop work in LangGraph?")

    print("\n Test 4: Out of context question ")
    ask("What is the capital of France?")  