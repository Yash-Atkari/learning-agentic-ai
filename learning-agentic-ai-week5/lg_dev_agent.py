import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Setup llm model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Define the tools
@tool
def check_emails(query: str) -> str:
    """Checks the emails. Returns a list of mock emails."""
    print(f"(Tool) Checking gmail for: {query}")
    return """
    1. [Promo] 50% off on React Course
    2. [Work] GitHub: Pull Request #42 needs review
    3. [Promo] Best Pizza in Nagpur
    4. [Work] Daily Standup Meeting Link
    """

@tool
def delete_promotions(confirm: bool) -> str:
    """Delete emails marked as [Promo]. Requires confirmation."""
    if confirm:
        print(f"(Tool) Deleting [Promo] emails from Gmail...")
        return "Success: Deleted 2 [Promo] emails."
    else:
        return "Action cancelled: User did not confirm."
    
@tool
def add_calendar_event(event_name: str, time: str) -> str:
    """Adds an event to the google calendar."""
    print(f"(Tool) Scheduling: '{event_name}' at '{time}'")
    return f"Success: Added '{event_name}' at '{time}' to google calendar."

@tool
def update_todo_list(task: str, action: str) -> str:
    """Updates the to-do list. Actions can be 'add' or 'remove'"""
    print(f"(Tool) To-Do List: {action.upper()} {task}")
    return f"Success: {action}ed {task} to your to-do list."

# Bind tools to the model
tools = [check_emails, delete_promotions, add_calendar_event, update_todo_list]
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
user_command = "Check my emails. If there are promos, delete them. Also, if there is a Standup meeting, add it to my calendar at 10 AM."

initial_state = {
    "messages": [
        SystemMessage(content="You are a smart assistant. You manage emails, calendar and tasks."),
        HumanMessage(content=user_command)
    ]
}

for event in app.stream(initial_state):
    for key, value in event.items():
        print(f"{key}'s turn:")
        last_msg = value["messages"][-1]
        if last_msg.content:
            print(f"Message: {last_msg.content}\n")
        else:
            print(f"Agent decided to call {len(last_msg.tool_calls)} tools.\n")
