import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# New imports for langgraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Setup model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Setup tools
@tool
def multiply(a: int, b: int) -> int:
    """Mulitiplies two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

tools = [multiply, add]
llm_with_tools = llm.bind_tools(tools)

# Define nodes
# In LangGraph, instead of a function with a loop, we define "Nodes"
def agent_node(state: MessagesState):
    """The Brain: Decides what to do next."""
    # We get the history from ''state'
    messages = state["messages"]

    # We ask the model
    response = llm_with_tools.invoke(messages)

    # We return the new message to update the 'state'
    return {"messages": [response]}

# The 'ToolNode' is pre-built! We don't need to write the execution loop manually.
tool_node = ToolNode(tools)

# Build the graph (The "Flowchart")
workflow = StateGraph(MessagesState)

# 1. Add the nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 2. Add the edges
workflow.add_edge(START, "agent") # 1. Start -> Go to Agent

# 3. Conditional Edge
# "If agent calls a tool, go to 'tools'. If agent is done, go to END."
# 'tools_condition' is a pre-built function that checks for tool_calls!
workflow.add_conditional_edges(
    "agent", 
    tools_condition
)

# 4. The Loop Back
# After tools run, go BACK to the agent (This creates the loop automatically)
workflow.add_edge("tools", "agent")

# 5. Compile the machine
app = workflow.compile()

# Run it
print("LangGraph Jarvis is Online...\n")

# We start with a System Message + User Message
initial_state = {
    "messages": [
        SystemMessage(content="You are Jarvis. You perform math calculations."),
        HumanMessage(content="Calculate 55 * 10, and then add 5 to the result.")
    ]
}

# Run the Graph
# stream() allows us to see each step happening
for event in app.stream(initial_state):
    for key, value in event.items():
        print(f"Current node: {key}")
        print(f"Message: {value['messages'][-1].content}\n")
