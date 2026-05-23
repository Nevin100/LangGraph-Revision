from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import sqlite3
from dotenv import load_dotenv

# Load the .env variables
load_dotenv()

# model 
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Node Function
def chatbot(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Graph 
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# SQLite Checkpointer 
# check_same_thread=False — multi-thread safe
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Compile 
graph = builder.compile(checkpointer=checkpointer)

# Streaming chat function 
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


# State inspect (bonus) 
def show_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.values:
        print("(no history found)")
        return
    for msg in state.values["messages"]:
        role = "You" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"{role}: {msg.content}\n")


# Main loop
if __name__ == "__main__":
    thread_id = "nevin-session-1" 

    print(f"Chat started [thread: {thread_id}]")
    print("Commands: 'history' — past msgs dekho | 'quit' — exit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        elif user_input == "quit":
            print("Bye!")
            break
        elif user_input == "history":
            print("\n Conversation History ")
            show_history(thread_id)
        else:
            chat(user_input, thread_id)