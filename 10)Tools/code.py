# Tools in LangGraph are functions that the LLM can choose to call on its own, based on the user's input.
# You define functions with the `@tool` decorator and bind them to the LLM using `llm.bind_tools(tools)`. The LLM reads each tool's docstring and decides at runtime whether to call a tool or answer directly.
# ToolNode is a built-in LangGraph node that automatically executes whichever tool the LLM decided to call and appends the result back to the messages list.
# tools_condition is a built-in conditional edge that checks whether the LLM made a tool call or not — if yes, it routes to the tools node; if no, it routes to END.
# The flow looks like this - LLM decides a tool is needed, ToolNode executes it, result goes back to the LLM, LLM gives the final response. If no tool is needed, LLM answers directly and the graph ends.

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file (like API keys, if needed)
load_dotenv()

# Tools definition  
# @tool decorator — is a way to define a function as a tool that the LLM can call. The function's docstring is crucial because the LLM reads it to understand what the tool does and when to use it.
# docstring is the description of the tool that helps the LLM understand its purpose and when to use it. A clear and concise docstring improves the LLM's ability to choose the right tool for a given user input.

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""
    # currently hardcoded data, but in a real app, you would call a weather API here
    weather_data = {
        "delhi": "32°C, sunny",
        "mumbai": "28°C, humid",
        "bangalore": "24°C, cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not found for {city}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2', '10 * 5'"""
    try:
        result = eval(expression)  # In a real app, use a safer parser
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_joke() -> str:
    """Tell a programming joke."""
    return "Why do programmers prefer dark mode? Because light attracts bugs!"


# Tools list 
tools = [get_weather, calculator, get_joke]

# LLM initialization and binding tools 
# bind_tools - is a method that connects the defined tools to the LLM. When you call `llm.bind_tools(tools)`, it allows the LLM to access and invoke these tools during the conversation based on the user's input and the tool's docstring.
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# Nodes 
def chatbot(state: MessagesState):
    """LLM node that generates responses and decides whether to call a tool or not based on the user's input and the tools' docstrings."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool Node - is a built-in LangGraph node that automatically executes whichever tool the LLM decided to call and appends the result back to the messages list. It simplifies the process of handling tool calls by abstracting away the execution logic, allowing you to focus on defining your tools and the overall conversation flow.
# ToolNode automatically detects which tool the LLM has chosen to call (based on the tool call format in the messages) and executes it, then appends the tool's output back to the messages list for the LLM to use in generating its final response.
tool_node = ToolNode(tools)

# Graph 
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")

# tools_condition is a built-in conditional edge that checks whether the LLM made a tool call or not — if yes, it routes to the tools node; if no, it routes to END. This allows the graph to dynamically decide the flow based on the LLM's decision to use a tool or not.
builder.add_conditional_edges("chatbot", tools_condition)

# If the LLM decides to call a tool, we route to the tools node. After the tool executes, we route back to the chatbot for the LLM to generate the final response using the tool's output.
builder.add_edge("tools", "chatbot")

graph = builder.compile()

# Chat function 
def chat(user_input: str):
    messages = [HumanMessage(content=user_input)]
    result = graph.invoke({"messages": messages})
    print(f"\nAssistant: {result['messages'][-1].content}")

# Main
if __name__ == "__main__":
    print("Test 1: Weather ")
    chat("Weather of Delhi?")

    print("\n Test 2: Calculator")
    chat("Calculate: 2455 * 869")

    print("\n Test 3: Joke ")
    chat("Tell me a programming joke.")

    print("\n Test 4: No tool needed")
    chat("What is Python") 