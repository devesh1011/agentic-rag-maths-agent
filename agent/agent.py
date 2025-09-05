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
from agent_guardrails import input_guard, output_guard
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


import dotenv
import asyncio
import os

dotenv.load_dotenv()

COLLECTION_NAME = "calculus_collection"
CHAT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"
KNOWLEDGE_BASE_PATH = "./langchain_qdrant"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class AgentState(TypedDict):
    """The state of the math routing agent"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    end: bool
    answer: str
    user_feedback: str


LLM = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)

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

mcp_client = MultiServerMCPClient(
    {
        "tavily-mcp": {
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
            "transport": "streamable_http",
        }
    }
)
all_tools = asyncio.run(mcp_client.get_tools())
tavily_search_tool = next(
    (tool for tool in all_tools if tool.name == "tavily_search"), None
)


@tool
def retriever_tool(query: str) -> str:
    """Search the calculus knowledge base for relevant information"""
    try:
        docs = retriever.invoke(query)
        if docs:
            solution_doc = docs[0]
            print(f"KB Result Metadata: {solution_doc.metadata}")
            return str(solution_doc.metadata)
        else:
            return "NO_RESULTS"
    except Exception as e:
        return "NO_RESULTS"


# Custom tool node to handle state updates
def custom_tool_node(state: AgentState):
    """Custom tool node that updates state with context"""
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
                    import asyncio

                    result = asyncio.run(tavily_search_tool.ainvoke(tool_args))
                    print(f"tavily search results:{result}")
                    tool_outputs.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    print(f"Error calling tavily search: {e}")
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

tools = [retriever_tool, tavily_search_tool]


def input_guardrails(state: AgentState):
    """
    Checks the user's input. If it's non-math, it ends the graph
    with a helpful message. Otherwise, it continues.
    """
    last_message = state["messages"][-1]
    original_content = last_message.content
    outcome = input_guard.validate(original_content)
    final_content = outcome.validated_output
    if final_content != original_content:
        state["messages"].append(AIMessage(content=final_content))
        state["end"] = True
    return state


def output_guardrails(state: AgentState):
    """
    Checks the agent's final response for quality, format, and safety.
    This acts as the "AI Gateway" before the response is shown to a human.
    """
    print("\n🔎 Running Output Guardrails (AI Gateway)...")
    last_message = state["messages"][-1]

    try:
        # Use .parse() to run validation. It will automatically handle the
        # OnFailAction (in this case, FIX) and return the result.
        validated_output = output_guard.parse(
            llm_output=last_message.content,
        )

        # Update the message with the validated (and possibly fixed) content.
        state["messages"][-1].content = validated_output.validated_output
        print("✅ Output Guardrails Passed.")

    except Exception as e:
        # This will catch any unexpected errors during the validation process.
        print(f"❌ Output Guardrails Failed with an exception: {e}")
        state["messages"][-1].content = (
            f"Output Guardrail Error: The agent's response failed validation.\n"
            f"Error: {e}"
        )

    return state


# Define the agent function that calls the LLM
def call_agent(state: AgentState):
    """Call the LLM with system prompt, context awareness, and current messages"""
    context = state.get("context", "")
    enhanced_prompt = system_prompt
    if context:
        enhanced_prompt += f"\n\n## [Knowledge Base Content]:\n{context}"
    messages = [SystemMessage(content=enhanced_prompt)] + state["messages"]
    response = LLM.bind_tools(tools).invoke(messages)
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
        # Agent is done, route to the output guardrail check first.
        return "output_guardrails"


def prepare_for_feedback(state: AgentState):
    """Populates the 'answer' field in the state from the last message."""
    last_message = state["messages"][-1]
    state["answer"] = last_message.content
    return state


def get_feedback(state: AgentState):
    """Pauses the graph to present the agent's answer and wait for human feedback."""
    print("--- Pausing for human feedback ---")
    human_input = interrupt({"answer_to_review": state["answer"]})
    return {"user_feedback": human_input}


builder = StateGraph(AgentState)

builder.add_node("agent", call_agent)
builder.add_node("tools", custom_tool_node)
builder.add_node("input_guardrails", input_guardrails)
builder.add_node("output_guardrails", output_guardrails)  # Keep this
builder.add_node("prepare_for_feedback", prepare_for_feedback)  # Keep this
builder.add_node("get_feedback", get_feedback)  # Keep this

# Define the graph's flow
builder.add_edge(START, "input_guardrails")
builder.add_conditional_edges(
    "input_guardrails", check_end_status, {"call_agent": "agent", "end_chat": END}
)

# The agent's decisions are now routed cleanly by should_continue
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "output_guardrails": "output_guardrails",  # This is the correct path
    },
)
builder.add_edge("tools", "agent")

# This is now a simple, linear path to the end
builder.add_edge("output_guardrails", "prepare_for_feedback")
builder.add_edge("prepare_for_feedback", "get_feedback")
builder.add_edge("get_feedback", END)

# Compile the graph
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# Create the agent
math_routing_agent = graph


def test_agent():
    """Test the math routing agent with the full human-in-the-loop flow."""
    test_queries = [
        # A valid math query to test the human-in-the-loop path
        r"Find the derivative of y = \frac{1}{24}(x^2 + 8)\sqrt{x^2 - 4} + \frac{x^2}{16}\arcsin\frac{2}{x}, x > 0",
        # # An invalid query to test the guardrail path
        # "What is the weather in Noida today?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")

        try:
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                context="",
                end=False,
                answer="",
                user_feedback="",
            )
            # A unique thread_id is required for each independent run
            config = {"configurable": {"thread_id": f"test-thread-{i}"}}

            print(f"\n🔄 Agent execution steps:")

            # Stream the graph. It will either run to completion (if a guardrail ends it)
            # or pause at the interrupt.
            for step_output in math_routing_agent.stream(initial_state, config=config):
                for node_name, node_output in step_output.items():
                    print(f"\n📍 Node: {node_name}")

            # After the stream is finished, get the final state to check how it ended.
            final_state_snapshot = math_routing_agent.get_state(config)
            final_state_values = final_state_snapshot.values

            # CORE FIX: Check if the graph was terminated by a guardrail.
            if final_state_values.get("end"):
                print("\n✅ Guardrail triggered. Agent run ended.")
                final_message = final_state_values["messages"][-1]
                print(f"\n🎯 Final Response:")
                print(f"  - {final_message.type.upper()}: {final_message.content}")
                # Use 'continue' to skip the feedback logic and move to the next query
                continue

            # If the graph didn't end, it must have paused for feedback.
            # Now we can safely proceed with the human intervention logic.
            proposed_answer = final_state_values["answer"]

            print("\n================== HUMAN INTERVENTION ==================")
            print(f"Agent's Proposed Answer:\n{proposed_answer}")
            print("==========================================================")

            feedback = input("Provide feedback and press Enter: ")

            print("\n🔄 Resuming agent execution...")
            # Resume the graph with the human's feedback.
            math_routing_agent.invoke(Command(resume=feedback), config=config)

            # Get the state one last time to show the complete, final result.
            final_run_state = math_routing_agent.get_state(config).values
            final_messages = final_run_state["messages"]
            user_feedback_captured = final_run_state["user_feedback"]

            print(f"\n🎯 Final Run State:")
            print("--- Full Conversation History ---")
            for msg in final_messages:
                print(f"  - {msg.type.upper()}: {msg.content}")
            print(f"  - HUMAN FEEDBACK CAPTURED: {user_feedback_captured}")
            print("---------------------------------")

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback

            traceback.print_exc()

        print(f"\n{'='*60}")


if __name__ == "__main__":
    print("🚀 Starting Math Routing Agent...")
    print("📚 Agent configured with:")
    print("  - Knowledge Base: Calculus problems collection")
    print("  - Web Search: Tavily MCP integration")
    print("  - Routing: Sequential (KB first, then web search)")
    test_agent()
