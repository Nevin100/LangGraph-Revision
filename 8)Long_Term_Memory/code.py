# Long term memory is the ability of a conversational agent to remember information across multiple threads/sessions.
# Important data is explicitly saved to a shared `Store` - any thread can read from it. Unlike short term memory, it is not bound to a `thread_id`, it is a global shared space accessible to everyone.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import sqlite3
from dotenv import load_dotenv

# Load environment variables (like API keys)
load_dotenv()

# Model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Long Term Store 
store = InMemoryStore()

#  Node
def chatbot(state: MessagesState, *, store: BaseStore):
    user_id = "nevin"  # real app mein dynamic hoga

    profile_item = store.get(("users", user_id), "profile")
    profile = profile_item.value if profile_item else {}

    system_msg = "You are a helpful assistant."
    if profile:
        system_msg += f" User details: {profile}"

    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = llm.invoke(messages)

    last_user_msg = state["messages"][-1].content.lower()

    if "mera naam" in last_user_msg:
        naam = last_user_msg.split("mera naam")[-1].strip().rstrip(".")
        current = profile or {}
        current["name"] = naam
        store.put(("users", user_id), "profile", current)
        print(f"[Memory Saved] naam: {naam}")

    if "main" in last_user_msg and "rehta" in last_user_msg:
        current = profile or {}
        current["location"] = last_user_msg
        store.put(("users", user_id), "profile", current)
        print(f"[Memory Saved] location info")

    return {"messages": [response]}

# Graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# SQLite checkpointer — long term memory peristance
conn = sqlite3.connect("long_term.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer, store=store)

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

# Main 
if __name__ == "__main__":
    print("Thread 1 — info passing in hindi ")
    chat("mera naam Nevin hai", thread_id="session-1")
    chat("main Delhi mein rehta hoon", thread_id="session-1")

    
    print("\nThread 2 — fresh thread, (but have memory) ")
    chat("mera naam kya hai?", thread_id="session-2")   
    chat("main kahan rehta hoon?", thread_id="session-2")  