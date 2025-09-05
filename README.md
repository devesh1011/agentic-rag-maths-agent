# Math Routing Agent - Complete Source Code Documentation

## 🎯 Project Overview

The Math Routing Agent is an advanced AI-powered mathematics tutoring system built using **Agentic-RAG (Retrieval-Augmented Generation)** architecture. The system intelligently routes math queries through a knowledge base first, then falls back to web search if needed, while incorporating comprehensive guardrails and human-in-the-loop feedback mechanisms.

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Math Routing Agent System                  │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)          │  Backend (FastAPI)            │
│  ├─ Chat Interface         │  ├─ Agent Orchestration       │
│  ├─ Feedback Collection    │  ├─ API Endpoints             │
│  └─ Markdown Rendering     │  └─ Session Management        │
├─────────────────────────────────────────────────────────────┤
│                    Core Agent (LangGraph)                   │
│  ├─ Input Guardrails       │  ├─ Output Guardrails         │
│  ├─ Knowledge Base Tool    │  ├─ Web Search Tool           │
│  ├─ Human-in-the-Loop     │  └─ Feedback Processing       │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                │  External Services            │
│  ├─ Qdrant Vector DB       │  ├─ Google Gemini LLM         │
│  ├─ Calculus Problems      │  ├─ Tavily Web Search         │
│  └─ Feedback Storage       │  └─ Google Embeddings         │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Table of Contents

1. [Core Agent Architecture](#core-agent-architecture)
2. [Component Deep Dive](#component-deep-dive)
3. [Data Flow and State Management](#data-flow-and-state-management)
4. [Guardrails System](#guardrails-system)
5. [Human-in-the-Loop Mechanism](#human-in-the-loop-mechanism)
6. [Knowledge Base Implementation](#knowledge-base-implementation)
7. [Web Interface](#web-interface)
8. [API Documentation](#api-documentation)
9. [Installation and Setup](#installation-and-setup)
10. [Usage Examples](#usage-examples)

---

## 🧠 Core Agent Architecture

### Agent State Definition (`AgentState`)

```python
class AgentState(TypedDict):
    """The state of the math routing agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    context: str                                              # Knowledge base results
    end: bool                                                # Guardrail termination flag
    answer: str                                              # Prepared answer for feedback
    user_feedback: str                                       # Human feedback input
```

The agent state serves as the central data structure that flows through all nodes in the LangGraph workflow, maintaining conversation context and routing decisions.

### LangGraph Workflow

The agent is built using **LangGraph's StateGraph** with the following node structure:

```python
# Node Definitions
builder.add_node("input_guardrails", input_guardrails)     # Input validation
builder.add_node("agent", call_agent)                      # LLM orchestration
builder.add_node("tools", custom_tool_node)                # Tool execution
builder.add_node("output_guardrails", output_guardrails)   # Output validation
builder.add_node("prepare_for_feedback", prepare_for_feedback)  # Feedback preparation
builder.add_node("get_feedback", get_feedback)             # Human interaction
```

### Workflow Flow Control

```python
# Conditional routing based on guardrails and tool calls
builder.add_conditional_edges(
    "input_guardrails",
    check_end_status,
    {"call_agent": "agent", "end_chat": END}
)

builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "output_guardrails": "output_guardrails"}
)
```

---

## 🔧 Component Deep Dive

### 1. Core Agent (`agent.py`)

#### **LLM Configuration**

```python
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,  # Low temperature for consistent mathematical answers
)
```

#### **System Prompt Design**

The agent uses a sophisticated system prompt that enforces the routing strategy:

```python
system_prompt = """
You are Professor MathAI, an expert math tutor. Your primary goal is to provide a step-by-step solution to the user's question using the tools provided.

## Core Instruction: Use the Knowledge Base First

Your primary and most trusted source of information is the [Knowledge Base Content]. This content is considered authoritative and pre-approved.

- **IF the [Knowledge Base Content] contains a full, step-by-step solution:** You MUST use that content and set source as "Knowledge Base"
- **IF the [Knowledge Base Content] is "NO_RESULTS":** Your ONLY next action is to call the `tavily_search` tool
- **After using `tavily_search`:** Use the web results to create the step-by-step solution

## Response Format:
📚 **Step-by-Step Solution**: [Clear, numbered steps]
💡 **Key Concepts**: [Mathematical principles used]
🔍 **Source**: [Knowledge Base or Web Search]
"""
```

#### **Tool Integration**

```python
@tool
def retriever_tool(query: str) -> str:
    """Search the calculus knowledge base for relevant information"""
    try:
        docs = retriever.invoke(query)
        if docs:
            solution_doc = docs[0]
            return str(solution_doc.metadata)  # Returns JSON metadata
        else:
            return "NO_RESULTS"
    except Exception as e:
        return "NO_RESULTS"
```

#### **Custom Tool Node**

The `custom_tool_node` handles both local retrieval and external web search:

```python
def custom_tool_node(state: AgentState):
    """Custom tool node that updates state with context"""
    initialize_mcp_tools()  # Lazy initialization of MCP tools

    # Handle retriever_tool calls
    if tool_name == "retriever_tool":
        result = retriever_tool.invoke(tool_args)
        state["context"] = result  # Store KB results in state

    # Handle tavily_search calls
    elif tool_name == "tavily_search":
        result = asyncio.run(tavily_search_tool.ainvoke(tool_args))

    return {"messages": tool_outputs}
```

### 2. Guardrails System (`agent_guardrails.py`)

#### **Input Guardrails - Math Content Detection**

```python
@register_validator(name="detect_non_math_content", data_type="string")
class NonMathContentDetector(Validator):
    def validate(self, value: Any, metadata: dict) -> ValidationResult:
        """Checks if the message contains legitimate math-related content."""

        messages = [
            SystemMessage(content=GUARD_PROMPT),
            HumanMessage(content=str(value)),
        ]

        response = guard_llm.invoke(messages)
        response = response.content.strip().upper()

        if "INVALID" in response:
            return FailResult(
                error_message="Non-math content detected.",
                fix_value="I'm sorry, I can only answer math-related questions.",
            )
        elif "VALID" in response:
            return PassResult()
```

The guardrail uses a dedicated LLM call to classify input as mathematical or non-mathematical content.

#### **Output Guardrails - Response Quality**

```python
def output_guardrails(state: AgentState):
    """AI Gateway for response validation"""
    last_message = state["messages"][-1]

    try:
        validated_output = output_guard.parse(llm_output=last_message.content)
        state["messages"][-1].content = validated_output.validated_output
    except Exception as e:
        state["messages"][-1].content = f"Output Guardrail Error: {e}"

    return state
```

### 3. Knowledge Base (`knowledge_base_qdrant.py`)

#### **Vector Store Configuration**

```python
# Qdrant configuration
COLLECTION_NAME = "calculus_collection"
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_SIZE = 768
VECTOR_DISTANCE = Distance.COSINE

# Vector store setup
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
client = QdrantClient(path=KNOWLEDGE_BASE_PATH)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Retriever configuration
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,                    # Return top 1 result
        "score_threshold": 0.7,    # Minimum similarity threshold
    },
)
```

#### **LaTeX Normalization**

```python
def normalize_latex(text):
    """Normalize LaTeX to reduce mismatches."""
    text = re.sub(r"\\cdot\s*", "", text)      # Remove \cdot
    text = re.sub(r"\\{2,}", r"\\", text)      # Fix double backslashes
    text = re.sub(r"\s+", " ", text.strip())   # Normalize spaces
    return text
```

### 4. Human-in-the-Loop Implementation

#### **Interrupt Mechanism**

```python
def get_feedback(state: AgentState):
    """Pauses the graph to present the agent's answer and wait for human feedback."""
    human_input = interrupt({"answer_to_review": state["answer"]})
    return {"user_feedback": human_input}
```

#### **Feedback Processing**

```python
def prepare_for_feedback(state: AgentState):
    """Populates the 'answer' field in the state from the last message."""
    last_message = state["messages"][-1]
    state["answer"] = last_message.content
    return state
```

The workflow pauses after the output guardrails, allowing humans to review and provide feedback before final completion.

---

## 🔄 Data Flow and State Management

### 1. Request Lifecycle

```
User Input → Input Guardrails → Agent (LLM) → Tool Selection → Tool Execution → Output Guardrails → Feedback Collection → Response
```

### 2. State Transitions

```python
# Initial state
AgentState(
    messages=[HumanMessage(content="Find derivative of x^2")],
    context="",
    end=False,
    answer="",
    user_feedback=""
)

# After knowledge base search
AgentState(
    messages=[...],
    context='{"solution": "...", "answer": "2x", "source": "Knowledge Base"}',
    end=False,
    answer="",
    user_feedback=""
)

# After feedback preparation
AgentState(
    messages=[...],
    context="...",
    end=False,
    answer="📚 **Step-by-Step Solution**: ...",
    user_feedback=""
)

# After human feedback
AgentState(
    messages=[...],
    context="...",
    end=False,
    answer="...",
    user_feedback="Looks good!"
)
```

### 3. Routing Logic

```python
def should_continue(state: AgentState):
    """Determine if we should continue to tools or to the output gateway."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # Execute tools
    else:
        return "output_guardrails"  # Move to validation
```

---

## 🛡️ Guardrails System

### Input Validation Strategy

1. **Math Content Detection**: Uses LLM-based classification to identify mathematical queries
2. **Profanity Filtering**: Implements content safety checks
3. **Early Termination**: Non-math queries are terminated with helpful messages

### Output Validation Strategy

1. **Format Compliance**: Ensures responses follow the required structure
2. **Content Safety**: Validates mathematical accuracy and appropriateness
3. **Quality Assurance**: Acts as an AI gateway before human presentation

### Guardrail Architecture

```python
# Input guardrails flow
def input_guardrails(state: AgentState):
    last_message = state["messages"][-1]
    original_content = last_message.content
    outcome = input_guard.validate(original_content)
    final_content = outcome.validated_output

    if final_content != original_content:
        state["messages"].append(AIMessage(content=final_content))
        state["end"] = True  # Terminate workflow
    return state
```

---

## 👥 Human-in-the-Loop Mechanism

### Implementation Using LangGraph Interrupts

The system uses LangGraph's `interrupt()` function to pause execution and collect human feedback:

```python
def get_feedback(state: AgentState):
    """Pauses the graph to present the agent's answer and wait for human feedback."""
    human_input = interrupt({"answer_to_review": state["answer"]})
    return {"user_feedback": human_input}
```

### Feedback Collection Workflow

1. **Answer Preparation**: Extract the agent's proposed solution
2. **Workflow Pause**: Use `interrupt()` to halt execution
3. **Human Review**: Present answer to human for evaluation
4. **Feedback Processing**: Resume with `Command(resume=feedback)`
5. **Final Response**: Complete the workflow with human input

### Resume Mechanism

```python
# In FastAPI endpoint
math_routing_agent.invoke(Command(resume=request.feedback), config=config)
```

---

## 💾 Knowledge Base Implementation

### Data Structure

The knowledge base contains 18,787 calculus problems stored in JSON format:

```json
{
  "problem": "Mathematical problem statement with LaTeX",
  "solution": "Detailed step-by-step solution",
  "answer": "Final answer in LaTeX format"
}
```

### Vector Database Setup

```python
# Qdrant collection initialization
def setup_qdrant_collection(client, collection_name, vector_size, distance):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
```

### Embedding Strategy

- **Model**: Google's `text-embedding-004`
- **Vector Size**: 768 dimensions
- **Distance Metric**: Cosine similarity
- **Retrieval Threshold**: 0.7 minimum similarity score

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8+
- Node.js 14+
- Google API Key (for Gemini LLM and embeddings)
- Tavily API Key (for web search)

### Backend Setup

```bash
# Navigate to agent directory
cd agent/

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_google_api_key
# TAVILY_API_KEY=your_tavily_api_key

# Initialize the knowledge base
python3 knowledge_base_qdrant.py

# Start the FastAPI server
python3 main.py
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/

# Install Node.js dependencies
npm install

# Start the React development server
npm start
```

### Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---
