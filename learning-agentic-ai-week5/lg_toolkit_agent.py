import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Toolkit imports
from langchain_google_community import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Setup llm model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Setup gmail toolkit
print("Initializing Gmail Toolkit...")
# This looks for 'credentials.json' in your folder automatically
toolkit = GmailToolkit()
# Get the list of pre-built tools (Read, Send, Search, etc.)
tools = toolkit.get_tools()

print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

# Define the graph node
def agent_node(state: MessagesState):
    """Decides next step based on history."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Prebuilt tool node
tool_node = ToolNode(tools)

# Build the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", tools_condition)

workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# Run it
user_command = "Check the last 3 emails in my Inbox that are NOT from me. Ignore sent items."

initial_state = {
    "messages": [
        SystemMessage(content="You are an advanced Email Assistant."),
        HumanMessage(content=user_command)
    ]
}

for event in app.stream(initial_state):
    for key, value in event.items():
        print(f"Current Node: {key}")
        last_msg = value["messages"][-1]
        if last_msg.content:
            print(f"Message: {last_msg.content}\n")
        else:
            print(f"Agent decided to call {len(last_msg.tool_calls)} tools.\n")
