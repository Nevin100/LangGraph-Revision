# Streaming in LangGraph:
# Streaming in LangGraph allows you to receive partial outputs from the LLM as they are generated, rather than waiting for the entire response. This can be particularly useful for improving user experience in real-time applications, as it reduces perceived latency and allows users to see responses as they come in.

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# State :
class State(BaseModel):
    messages: Annotated[list, add_messages]
    processed_input: str = ""

# LLM instance with streaming enabled
llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True) 
# streaming enabled helps to get partial responses as they are generated, which is useful for real-time applications and can improve user experience by reducing perceived latency.

# Nodes
def preprocess(state: State) -> dict:
    last_msg = state.messages[-1].content.strip()
    return {"processed_input": last_msg.lower()}

async def call_llm(state: State) -> dict:
    response = await llm.ainvoke(state.messages)
    return {"messages": [response]}

def postprocess(state: State) -> dict:
    # e.g. log, format, store — for now just passthrough
    return {}

# Graph 
builder = StateGraph(State)

# Adding Nodes and Edges
builder.add_node("preprocess", preprocess)
builder.add_node("call_llm", call_llm)
builder.add_node("postprocess", postprocess)

builder.add_edge(START, "preprocess")
builder.add_edge("preprocess", "call_llm")
builder.add_edge("call_llm", "postprocess")
builder.add_edge("postprocess", END)

# Compile the graph
graph = builder.compile()