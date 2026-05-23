# Human in the Loop (HITL) is a design pattern where a graph pauses mid-execution to wait for a human decision before continuing.
# The graph runs until it hits `interrupt()`, saves its entire state to the checkpointer, and waits. The human reviews the output and provides a decision. Then `Command(resume=value)` is called — the graph loads its saved state and continues from exactly where it paused.
# Checkpointer is mandatory - without it the state cannot be saved during the pause, so resume is impossible.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
from dotenv import load_dotenv

# Load the .env file to access environment variables (like API keys)
load_dotenv()

# Model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Node Functions
def draft_email(state: MessagesState):
    print("\n[Node 1] AI email is waiting..")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def human_review(state: MessagesState):
    draft = state["messages"][-1].content
    print(f"\n[Node 2] Draft is ready:\n{draft}\n")

    decision = interrupt({
        "question": "should we forward the mail?",
        "draft": draft
    })

    if decision == "approve":
        return {"messages": [AIMessage(content=f"Email approved and sent!\n\n{draft}")]}
    else:
        return {"messages": [AIMessage(content=f"Email cancelled. Feedback: {decision}")]}


def send_email(state: MessagesState):
    print("\n[Node 3] Email Sent!")
    return state

# Graph 
builder = StateGraph(MessagesState)
builder.add_node("draft_email", draft_email)
builder.add_node("human_review", human_review)
builder.add_node("send_email", send_email)

builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "human_review")
builder.add_edge("human_review", "send_email")
builder.add_edge("send_email", END)

# Checkpointer
conn = sqlite3.connect("hitl.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

# Execution 
if __name__ == "__main__":
    thread_id = "hitl-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    # Step 1: Graph Starting
    print("Step 1: Graph start") 
    user_message = HumanMessage(
        content="Draft a professional and formal email to my manager asking for a python developer position."
    )

    result = graph.invoke({"messages": [user_message]}, config=config)

    # Step 2: Current state View
    state = graph.get_state(config)
    print(f"\n[State] Next node: {state.next}")

    # Step 3: Human decision 
    print("\n Step 3: Human Decision") 
    human_decision = input("Approve ? (approve / reject with feedback): ").strip()

    # Step 4: Resume the Command 
    final_result = graph.invoke(
        Command(resume=human_decision),
        config=config
    )

    print("\n Final Result ")
    print(final_result["messages"][-1].content)