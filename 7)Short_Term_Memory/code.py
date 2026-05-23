# Short term memory is the ability of a conversational agent to remember messages within a single thread/session only.
# Each message is appended to a `messages` list and persisted via a checkpointer - but only accessible to that specific `thread_id`. Different thread = no memory.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import sqlite3
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

def chatbot(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

conn = sqlite3.connect("short_term.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)

def chat(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    messages = [HumanMessage(content=user_input)]

    print("\nAssistant: ", end="", flush=True)
    for chunk, metadata in graph.stream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and chunk.content:
            if chunk.__class__.__name__ == "AIMessageChunk":
                print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    print("Thread 1 ")
    chat("mera naam Nevin hai", thread_id="session-1")
    chat("main kya padh raha hoon? (hint: maine bataya tha)", thread_id="session-1")

    chat("mera naam kya hai?", thread_id="session-1")  

    print("\n Thread 2 (fresh start) ")
    chat("mera naam kya hai?", thread_id="session-2")  
