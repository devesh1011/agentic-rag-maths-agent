from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Sequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langgraph.graph import add_messages
import sys
import os
from agent_guardrails import input_guard, output_guard
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

import dotenv
import asyncio
import os

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_DIR)


dotenv.load_dotenv()

COLLECTION_NAME = "calculus_collection"
CHAT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"
KNOWLEDGE_BASE_PATH = os.path.join(AGENT_DIR, "langchain_qdrant")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# Initialize the state
class AgentState(TypedDict):
    """The state of the math routing agent"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    end: bool
    answer: str
    user_feedback: str


# Initalize the LLM
LLM = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)

# qdrant vector store initalized
client = QdrantClient(path=KNOWLEDGE_BASE_PATH)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.7,
    },
)

# Initialize MCP client
mcp_client = MultiServerMCPClient(
    {
        "tavily-mcp": {
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
            "transport": "streamable_http",
        }
    }
)


all_tools = None
tavily_search_tool = None


def initialize_mcp_tools():
    """Initialize MCP tools if not already done"""
    global all_tools, tavily_search_tool
    if tavily_search_tool is None:
        try:
            import nest_asyncio

            nest_asyncio.apply()
            all_tools = asyncio.run(mcp_client.get_tools())
            tavily_search_tool = next(
                (tool for tool in all_tools if tool.name == "tavily_search"), None
            )
        except ImportError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    tavily_search_tool = None
                else:
                    all_tools = asyncio.run(mcp_client.get_tools())
                    tavily_search_tool = next(
                        (tool for tool in all_tools if tool.name == "tavily_search"),
                        None,
                    )

            except RuntimeError:
                all_tools = asyncio.run(mcp_client.get_tools())
                tavily_search_tool = next(
                    (tool for tool in all_tools if tool.name == "tavily_search"), None
                )

        except Exception as e:

            tavily_search_tool = None


# vector DB retrieval tool
@tool
def retriever_tool(query: str) -> str:
    """Search the calculus knowledge base for relevant information"""
    try:
        docs = retriever.invoke(query)
        if docs:
            solution_doc = docs[0]

            return str(solution_doc.metadata)
        else:
            return "NO_RESULTS"
    except Exception as e:
        return "NO_RESULTS"


# Custom tool node to handle state updates
def custom_tool_node(state: AgentState):
    """Custom tool node that updates state with context"""
    initialize_mcp_tools()

    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "retriever_tool":
                result = retriever_tool.invoke(tool_args)
                tool_outputs.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

                state["context"] = result

            elif tool_name == "tavily_search":
                try:
                    if tavily_search_tool:
                        result = asyncio.run(tavily_search_tool.ainvoke(tool_args))

                        tool_outputs.append(
                            ToolMessage(content=result, tool_call_id=tool_call["id"])
                        )
                    else:
                        result = "Web search tool not available"
                        tool_outputs.append(
                            ToolMessage(content=result, tool_call_id=tool_call["id"])
                        )
                except Exception as e:

                    result = f"Web search failed: {e}"
                    tool_outputs.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )

        return {"messages": tool_outputs}

    return {"messages": []}


# System prompt for the agent
system_prompt = """
You are Professor MathAI, an expert math tutor. Your primary goal is to provide a step-by-step solution to the user's question using the tools provided.

## Core Instruction: Use the Knowledge Base First

Your primary and most trusted source of information is the [Knowledge Base Content]. This content is considered authoritative and pre-approved.

- **IF the [Knowledge Base Content] contains a full, step-by-step solution that directly answers the user's question:** You MUST use that content to formulate your final answer. You will then set the source as "Knowledge Base" and you will NOT call any other tools.

- **IF the [Knowledge Base Content] is "NO_RESULTS" or does not contain the solution:** Your ONLY next action is to call the `tavily_search` tool to find the information.

- **After using `tavily_search`:** Use the web results to create the step-by-step solution.

You must follow these rules without deviation.

## Response Format (ONLY for the final answer):
📚 **Step-by-Step Solution**: [Provide clear, numbered steps for the solution.]
💡 **Key Concepts**: [List the mathematical principles or rules used (e.g., Product Rule, Chain Rule).]
🔍 **Source**: [State which tool provided the information: "Knowledge Base" or "Web Search".]
"""

tools = [retriever_tool]


def input_guardrails(state: AgentState):
    """
    Checks the user's input. If it's non-math, it ends the graph
    with a helpful message. Otherwise, it continues.
    """
    last_message = state["messages"][-1]
    original_content = last_message.content
    outcome = input_guard.validate(original_content)
    if not outcome.is_valid:
        # Use fixed_value from our custom ValidationResult
        final_content = outcome.fixed_value
        state["messages"].append(AIMessage(content=final_content))
        state["end"] = True
    return state


def output_guardrails(state: AgentState):
    """
    Checks the agent's final response for quality, format, and safety.
    This acts as the "AI Gateway" before the response is shown to a human.
    """
    last_message = state["messages"][-1]

    try:
        # Use our custom validation instead of parse method
        validation_result = output_guard.validate(last_message.content)

        if not validation_result.is_valid:
            state["messages"][-1].content = validation_result.fixed_value
        # If validation passes, keep the original content

    except Exception as e:
        state["messages"][-1].content = (
            f"Output Guardrail Error: The agent's response failed validation.\n"
            f"Error: {e}"
        )

    return state


# Defining the agent to call the llm with the agntstate
def call_agent(state: AgentState):
    """Call the LLM with system prompt, context awareness, and current messages"""
    # Initialize MCP tools if not already done
    initialize_mcp_tools()

    context = state.get("context", "")
    enhanced_prompt = system_prompt
    if context:
        enhanced_prompt += f"\n\n## [Knowledge Base Content]:\n{context}"
    messages = [SystemMessage(content=enhanced_prompt)] + state["messages"]

    # Use tools that are available
    available_tools = [retriever_tool]
    if tavily_search_tool:
        available_tools.append(tavily_search_tool)

    response = LLM.bind_tools(available_tools).invoke(messages)
    return {"messages": [response]}


# input guard condition
def check_end_status(state: AgentState):
    if state.get("end", False):
        return "end_chat"
    else:
        return "call_agent"


def should_continue(state: AgentState):
    """Determine if we should continue to tools or to the output gateway."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        # For API usage, skip feedback and go directly to end
        return "output_guardrails"


def prepare_for_feedback(state: AgentState):
    """Populates the 'answer' field in the state from the last message."""
    last_message = state["messages"][-1]
    state["answer"] = last_message.content
    return state


def get_feedback(state: AgentState):
    """For API usage, skip feedback and return the answer directly."""
    # Instead of interrupting, just return the answer
    return {"user_feedback": "auto_approved"}


# Bulding the Workflow
builder = StateGraph(AgentState)

builder.add_node("agent", call_agent)
builder.add_node("tools", custom_tool_node)
builder.add_node("input_guardrails", input_guardrails)
builder.add_node("output_guardrails", output_guardrails)
builder.add_node("prepare_for_feedback", prepare_for_feedback)
builder.add_node("get_feedback", get_feedback)

# graph flow
builder.add_edge(START, "input_guardrails")
builder.add_conditional_edges(
    "input_guardrails", check_end_status, {"call_agent": "agent", "end_chat": END}
)

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "output_guardrails": "output_guardrails",
    },
)
builder.add_edge("tools", "agent")

builder.add_edge("output_guardrails", "prepare_for_feedback")
builder.add_edge("prepare_for_feedback", "get_feedback")
builder.add_edge("get_feedback", END)

# Compile the graph
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# Create the agent
math_routing_agent = graph
