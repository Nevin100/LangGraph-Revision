# LangGraph Workflow Patterns & Advanced Topics

A comprehensive learning repository demonstrating LangGraph's core workflow patterns and advanced features for building AI-powered applications with LLM agents.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Project Structure](#project-structure)
4. [Core Workflow Patterns](#core-workflow-patterns)
5. [Key Concepts & Implementations](#key-concepts--implementations)
6. [Advanced Topics](#advanced-topics)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [Dependencies](#dependencies)
9. [Contributing & Future Work](#contributing--future-work)

---

## Project Overview

This repository serves as an educational resource for understanding **LangGraph**, a framework for building stateful multi-actor applications with language models. The project demonstrates three fundamental workflow patterns:

- **Sequential Workflows** - Linear execution of tasks in order
- **Parallel Workflows** - Concurrent execution of independent tasks
- **Conditional Workflows** - Dynamic routing based on input classification

Each pattern includes both simple demonstrations and advanced LLM-integrated implementations using the Groq API.

**Target Audience:** Developers learning LangGraph, AI engineers building agent systems, and teams implementing multi-step AI workflows.

---

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Groq API key (for LLM features)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd langgraph-revision
   ```

2. **Create virtual environment (if not already done):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env  # Create .env file if needed
   # Add your GROQ_API_KEY to .env
   export GROQ_API_KEY="your-api-key-here"
   ```

5. **Run notebooks:**
   ```bash
   jupyter notebook
   ```

---

## Project Structure

```
langgraph-revision/
├── 1)Sequential_Workflow/
│   ├── workflow.ipynb          # Basic sequential workflow
│   └── llmworkflow.ipynb       # Sequential workflow with LLM integration
│
├── 2)Parallel_Workflow/
│   ├── simple_workflow.ipynb   # Basic parallel execution
│   └── llm_workflow.ipynb      # Parallel LLM tasks with Annotated types
│
├── 3)Conditional_Workflows/
│   ├── simple_workflow.ipynb   # Conditional routing (quadratic solver)
│   └── llm_workflow.ipynb      # LLM-based conditional router
│
├── main.py                      # Main entry point
├── requirements.txt             # Project dependencies
├── pyproject.toml              # Project metadata
└── README.md                   # This file
```

---

## Core Workflow Patterns

### 1. Sequential Workflows

**Concept:** Tasks execute one after another in a strict order. Output of one task becomes input to the next.

**Characteristics:**
- Linear execution path
- Each node waits for previous node to complete
- State flows sequentially through nodes

**Real-world Use Cases:**
- Document processing pipelines
- Data transformation chains
- Step-by-step problem solving

**Key Files:**
- `1)Sequential_Workflow/workflow.ipynb` - Simple example with static data
- `1)Sequential_Workflow/llmworkflow.ipynb` - LLM integration for multi-step reasoning

**Example Flow:**
```
Input → Node1 → Node2 → Node3 → Output
```

---

### 2. Parallel Workflows

**Concept:** Multiple independent tasks execute simultaneously. Results are aggregated when all tasks complete.

**Characteristics:**
- Concurrent execution of multiple nodes
- Requires `Annotated` types for state fields receiving parallel updates
- Better performance for independent tasks
- Reduces total execution time

**Real-world Use Cases:**
- Multi-perspective analysis (summarize, extract key points, suggest actions simultaneously)
- Parallel data enrichment
- Independent calculations that need aggregation

**Key Files:**
- `2)Parallel_Workflow/simple_workflow.ipynb` - Basic parallel execution
- `2)Parallel_Workflow/llm_workflow.ipynb` - LLM tasks running in parallel

**Important:** Using `Annotated[type, ""]` prevents `InvalidUpdateError` when parallel nodes update the same state field.

**Example Flow:**
```
       ┌─→ Node1 ─┐
Input →┼─→ Node2 ─┼→ Aggregator → Output
       └─→ Node3 ─┘
```

---

### 3. Conditional Workflows

**Concept:** Workflow path is determined dynamically based on routing logic. Different branches execute based on conditions.

**Characteristics:**
- Decision-based routing
- Router node returns destination node name (as string)
- Enables dynamic behavior based on input analysis
- Efficient filtering of tasks

**Real-world Use Cases:**
- Intent-based routing (customer support, question classification)
- Specialized processing pipelines (math vs. general questions)
- Dynamic task selection based on input type

**Key Files:**
- `3)Conditional_Workflows/simple_workflow.ipynb` - Quadratic equation solver with conditional branches
- `3)Conditional_Workflows/llm_workflow.ipynb` - LLM-based question classifier and responder

**Example Flow:**
```
Input → Router → Decision: "math" or "general"?
           ↓                    ↓
         Math Node          General Node
           ↓                    ↓
        Output             Output
```

---

## Key Concepts & Implementations

### State Management

**TypedDict Definition:**
```python
class WorkflowState(TypedDict):
    question: str
    category: str
    answer: str
```

States define all data flowing through the workflow. LangGraph maintains state consistency across nodes.

### Node Functions

Nodes are functions that process state and return updated state:

```python
def math_node(state: WorkflowState) -> WorkflowState:
    """Process math questions"""
    response = llm.invoke(prompt)
    state['answer'] = response.content  # Extract content, not entire object
    return state
```

**Important:** Always extract `.content` from LLM responses, not the entire message object.

### Graph Construction

LangGraph uses a builder pattern to construct workflows:

```python
builder = StateGraph(WorkflowState)
builder.add_node("input_node", input_node)
builder.add_edge(START, "input_node")
builder.add_conditional_edges("input_node", router_function)
workflow = builder.compile()
```

### Router Functions

Router functions determine which node to execute next:

```python
def llm_router(state: WorkflowState) -> Literal['math_node', 'general_node']:
    """Classify and route to appropriate node"""
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    state["category"] = category
    return category + "_node"  # Return node name as STRING
```

**Critical:** Router must return node name as string, not execute the node.

### Handling Parallel Updates with Annotated

When parallel nodes update the same state field:

```python
from typing import Annotated

class WorkflowState(TypedDict):
    summary: Annotated[str, ""]        # Metadata tells LangGraph how to merge
    key_points: Annotated[str, ""]
    action_items: Annotated[str, ""]
```

The `Annotated` metadata allows safe concurrent updates. Without this, `InvalidUpdateError` occurs.

---

## Advanced Topics

### 1. Iterative Workflows

**Definition:** Workflows that repeat nodes or loops until a condition is met.

**Use Cases:**
- Refinement loops (draft → review → refine)
- Validation with retry logic
- Agent decision-making loops

**Implementation Approach:**
```python
# Add conditional edge back to same node for iteration
graph.add_conditional_edges("review_node", should_refine)
# Router returns same node for iteration or next node to continue
```

**Key Patterns:**
- **Retry Logic:** Attempt operation, check success, retry if failed
- **Refinement Loops:** Generate → Evaluate → Refine → Repeat
- **Agentic Loops:** Agent thinks → acts → observes → repeats until done

---

### 2. Human-in-the-Loop (HITL)

**Definition:** Workflows that pause for human input/approval before proceeding.

**Use Cases:**
- Approval workflows (high-stakes decisions need human review)
- Interactive debugging
- User confirmation for critical actions
- Feedback collection and incorporation

**Implementation Approach:**
```python
def human_review_node(state: WorkflowState) -> WorkflowState:
    """Pause and wait for human input"""
    # Get human feedback (from UI, API, etc.)
    human_feedback = input("Approve this action? (y/n): ")
    state['human_decision'] = human_feedback
    return state

# Add conditional edge based on human decision
graph.add_conditional_edges("review_node", route_based_on_human_input)
```

**Considerations:**
- Requires external API/UI for user interaction
- State must be persistent to survive interruptions
- Need clear feedback mechanisms

---

### 3. Persistence & State Management

**Definition:** Saving workflow state to allow resumption after interruptions.

**Use Cases:**
- Long-running workflows that might fail
- Audit trails and compliance
- Workflow history and debugging
- Multi-session conversations

**Implementation Approach:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
# or for persistent storage:
# memory = SqliteSaver.from_conn_string("file:./workflow.db")

workflow = builder.compile(checkpointer=memory)

# Resume workflow from checkpoint
result = workflow.invoke(
    initial_state,
    config={"configurable": {"thread_id": "user-123"}}
)
```

**Persistence Strategies:**
- **In-Memory:** Fast but lost on restart
- **SQLite:** Good for development and testing
- **PostgreSQL:** Scalable for production
- **Custom Storage:** Cloud databases, etc.

---

### 4. Streaming in LangGraph

**Definition:** Real-time output streaming as workflow executes, instead of waiting for completion.

**Use Cases:**
- Large LLM outputs (show tokens as they generate)
- User experience improvement (no waiting for full response)
- Progressive results display
- Real-time monitoring

**Implementation Approach:**
```python
# Stream entire workflow values
for event in workflow.stream(initial_state):
    print(event)

# Stream specific node outputs
for event in workflow.stream(initial_state, mode="updates"):
    print(event)

# Stream with mode="values" for state snapshots
for event in workflow.stream(initial_state, mode="values"):
    print(event)
```

**Benefits:**
- Better user experience
- Earlier error detection
- Real-time debugging

---

### 5. Memory Systems in LangGraph

#### Short-Term Memory

**Definition:** Conversation context within current session.

**Implementation:**
```python
class ConversationState(TypedDict):
    messages: list  # Chat history
    current_message: str
    response: str
```

**Use Cases:**
- Multi-turn conversations
- Context-aware responses
- Temporary working memory

**Example:**
```python
def chat_node(state: ConversationState) -> ConversationState:
    # Include conversation history in prompt
    context = "\n".join([f"{msg['role']}: {msg['content']}" 
                         for msg in state['messages']])
    prompt = f"Conversation:\n{context}\n\nRespond to: {state['current_message']}"
    response = llm.invoke(prompt)
    
    # Add to history
    state['messages'].append({"role": "user", "content": state['current_message']})
    state['messages'].append({"role": "assistant", "content": response.content})
    return state
```

#### Long-Term Memory

**Definition:** Persistent storage of knowledge/information across sessions.

**Implementation Approaches:**

1. **Vector Database (RAG):**
   ```python
   # Store important facts in embeddings
   embeddings = create_embeddings(important_facts)
   vector_db.add(embeddings)
   
   # Retrieve relevant facts in workflow
   relevant_context = vector_db.search(query)
   ```

2. **Knowledge Base:**
   ```python
   # Store structured information
   knowledge = {
       "user_preferences": {...},
       "learned_patterns": {...},
       "historical_data": {...}
   }
   ```

3. **Database Storage:**
   ```python
   # Use SQLite, PostgreSQL, or MongoDB
   # Store summaries, insights, user profiles
   ```

**Use Cases:**
- User preferences and history
- Learned patterns from past interactions
- Knowledge base for domain-specific information
- Multi-session conversation continuity

---

### 6. LangSmith Integration & Monitoring

**Definition:** Powerful tool for debugging, testing, and monitoring LangGraph workflows.

**Key Features:**
- **Tracing:** See complete workflow execution trace
- **Debugging:** Inspect state at each node
- **Testing:** Run test cases and track results
- **Monitoring:** Track performance and errors
- **Feedback:** Collect and analyze user feedback

**Setup:**
```bash
# Install LangSmith client
pip install langsmith

# Set environment variables
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="langgraph-revision"
```

**Usage in Code:**
```python
from langsmith import traceable

@traceable
def my_node(state: WorkflowState) -> WorkflowState:
    # Your node logic
    return state

# Automatic tracing to LangSmith
result = workflow.invoke(initial_state)
```

**Key Benefits:**
- Visual trace of entire workflow execution
- Performance metrics and bottleneck identification
- Error tracking and debugging
- Input/output visibility at each step
- Cost analysis (LLM calls, tokens used)

**LangSmith Dashboard Features:**
- **Runs:** View each workflow execution
- **Traces:** Drill down into individual node executions
- **Datasets:** Test workflows with curated test cases
- **Feedback:** Tag runs as good/bad for analysis
- **Analytics:** Aggregate metrics and performance trends

---

## Common Issues & Solutions

### Issue 1: InvalidUpdateError - "unhashable type: 'dict'"

**Symptom:**
```
TypeError: unhashable type: 'dict'
At key 'summary': Can receive only one value per step
```

**Cause:** Multiple parallel nodes updating same state field without `Annotated` type.

**Solution:**
```python
# ❌ Wrong:
class WorkflowState(TypedDict):
    summary: str

# ✅ Correct:
class WorkflowState(TypedDict):
    summary: Annotated[str, ""]
```

---

### Issue 2: Router Returning Node Object Instead of String

**Symptom:**
```
TypeError: unhashable type: 'dict'
```

**Cause:** Router function executes node and returns dict instead of returning node name.

**Solution:**
```python
# ❌ Wrong:
def router(state):
    if condition:
        return node_function(state)  # Returns dict

# ✅ Correct:
def router(state):
    if condition:
        return "node_name"  # Return string node name
```

---

### Issue 3: LLM Response Appears as Object

**Symptom:**
```
Output: <langchain_core.messages.ai_message.AIMessage object at 0x...>
```

**Cause:** Not extracting `.content` from LLM response.

**Solution:**
```python
# ❌ Wrong:
state['answer'] = response

# ✅ Correct:
state['answer'] = response.content
```

---

### Issue 4: Missing Environment Variables

**Symptom:**
```
AuthenticationError: Groq API key not found
```

**Solution:**
```bash
# Add to .env file or environment:
export GROQ_API_KEY="your-actual-api-key"

# Or in Python:
from dotenv import load_dotenv
load_dotenv()
```

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **langgraph** | >=1.2.0 | Workflow orchestration framework |
| **langchain** | >=1.3.1 | LLM integration and tools |
| **langchain-groq** | >=1.1.2 | Groq LLM API integration |
| **python-dotenv** | >=1.2.2 | Environment variable management |
| **ipython** | >=9.13.0 | Interactive notebook environment |

### Installation via Requirements

```bash
pip install -r requirements.txt
```

### Installation via Pyproject.toml

```bash
pip install -e .
```

---

## Code Examples & Patterns

### Example 1: Running a Simple Sequential Workflow

```python
from typing import TypedDict
from langgraph.graph import START, END, StateGraph

class SimpleState(TypedDict):
    text: str
    processed: str

def step1(state: SimpleState) -> SimpleState:
    state['processed'] = state['text'].upper()
    return state

def step2(state: SimpleState) -> SimpleState:
    state['processed'] += "!"
    return state

builder = StateGraph(SimpleState)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

workflow = builder.compile()
result = workflow.invoke({"text": "hello", "processed": ""})
```

---

### Example 2: Parallel Workflows with Aggregation

```python
from typing import Annotated
from operator import add

class ParallelState(TypedDict):
    query: str
    results: Annotated[list, add]  # Accumulates results

def search1(state: ParallelState) -> ParallelState:
    return {"results": ["Result from source 1"]}

def search2(state: ParallelState) -> ParallelState:
    return {"results": ["Result from source 2"]}

builder = StateGraph(ParallelState)
builder.add_node("search1", search1)
builder.add_node("search2", search2)
builder.add_edge(START, ["search1", "search2"])  # Parallel
builder.add_edge(["search1", "search2"], END)

workflow = builder.compile()
result = workflow.invoke({"query": "test", "results": []})
```

---

### Example 3: LLM-Powered Conditional Router

```python
from langchain_groq import ChatGroq
from typing import Literal

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def classify_query(state) -> Literal['math_path', 'general_path']:
    prompt = f"Is this a math question? Reply only: 'math' or 'general'. Query: {state['query']}"
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    return "math_path" if category == "math" else "general_path"

# Add to graph
builder.add_conditional_edges("classifier", classify_query)
```

---

## Learning Path

### Beginner (Weeks 1-2)
1. Understand graph concepts (nodes, edges)
2. Run simple sequential workflow examples
3. Modify examples to understand state flow
4. Read documentation on basic patterns

### Intermediate (Weeks 3-4)
1. Implement parallel workflows
2. Debug `Annotated` type issues
3. Build conditional routing logic
4. Integrate with LLM APIs
5. Understand state management

### Advanced (Weeks 5-8)
1. Implement HITL workflows
2. Add persistence and checkpointing
3. Integrate LangSmith for monitoring
4. Build iterative refinement loops
5. Implement memory systems
6. Optimize performance

### Expert (Weeks 9+)
1. Production deployment patterns
2. Multi-agent workflows
3. Custom memory backends
4. Advanced streaming strategies
5. Complex error handling

---

## Next Steps & Future Development

### Planned Enhancements

1. **Iterative Workflows Example**
   - Implement draft → review → refine loop
   - Show retry logic and error handling
   - Demonstrate agent thought patterns

2. **Human-in-the-Loop Implementation**
   - Build approval workflow example
   - Create UI for human feedback
   - Show state persistence across interruptions

3. **Persistence Layer**
   - Add SQLite checkpointer examples
   - Show multi-session continuity
   - Demonstrate workflow resumption

4. **Streaming Examples**
   - Real-time token streaming from LLM
   - Progressive result display
   - WebSocket integration example

5. **Memory Systems**
   - Implement RAG (Retrieval Augmented Generation)
   - Build conversation memory system
   - Create long-term knowledge base

6. **LangSmith Integration**
   - Full monitoring setup
   - Custom metrics tracking
   - Performance optimization guide

### Contributing

To contribute examples or improvements:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Add well-documented examples
4. Test thoroughly
5. Submit pull request with documentation

---

## References & Resources

### Official Documentation
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs)
- [Groq API Documentation](https://groq.com/docs)

### Key Concepts
- [State Graphs in LangGraph](https://python.langchain.com/docs/langgraph/concepts/low_level_conceptual_guide)
- [Routing and Conditional Edges](https://python.langchain.com/docs/langgraph/how-tos/define-nodes-edges)
- [Persistence and Checkpointing](https://python.langchain.com/docs/langgraph/concepts/persistence)

### Related Tools
- [LangSmith - Debugging & Monitoring](https://smith.langchain.com/)
- [LangChain Templates](https://templates.langchain.com/)
- [LangGraph Hub](https://langchain-ai.github.io/langgraph/reference/library/)

---

## Troubleshooting Guide

### Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| `ModuleNotFoundError: No module named 'langgraph'` | Run `pip install -r requirements.txt` |
| `AuthenticationError` with LLM | Check `GROQ_API_KEY` environment variable |
| Workflow runs but produces no output | Check node return statements are updating state |
| Parallel execution fails with state error | Use `Annotated` for concurrent state updates |
| Router function not working | Ensure router returns string (node name), not object |

---

## FAQ

**Q: What's the difference between parallel and sequential workflows?**  
A: Sequential executes nodes one-by-one; parallel executes independent nodes simultaneously, reducing total time.

**Q: When should I use conditional edges?**  
A: When different paths should execute based on runtime conditions or input classification.

**Q: How do I persist workflow state?**  
A: Use LangGraph's checkpointer (SQLite, PostgreSQL, etc.) to save state at each step.

**Q: Can I use LangGraph without LLMs?**  
A: Yes, LangGraph works with any Python functions, though LLM integration is powerful.

**Q: What's the purpose of `Annotated` in state definitions?**  
A: It tells LangGraph how to merge values when multiple nodes update the same field concurrently.

---

## License

This project is provided as an educational resource.

---

## Contact & Support

For questions or issues:
- Check the troubleshooting section
- Review example notebooks
- Consult official LangGraph documentation

---

**Last Updated:** May 2024  
**Repository Version:** 0.1.0
