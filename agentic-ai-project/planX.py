import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Toolkit imports
from langchain_google_community import GmailToolkit, CalendarToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

system_prompt = """
You are planX, a smart assistant for a Developer/Student.
Your Goal: Automate daily tasks to save time.

Capabilities:
1. GMAIL: Check unread emails, send replies, delete promotions.
2. CALENDAR: Check schedule, book meetings, find free slots.

Guidelines:
- If a user asks to "start my day", check BOTH emails and calendar.
- Always confirm before sending an email or deleting items.
- Be concise. Developers don't like long paragraphs.
"""
print("PlanX system initializing...")

# Setup llm model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Setup gmail toolkit
print("   - Loading Gmail Toolkit...")
# This looks for 'credentials.json' in your folder automatically
gmail_tools = GmailToolkit().get_tools()
print("   - Loading Calendar Toolkit...")
# Get the list of pre-built tools (Read, Send, Search, etc.)
calendar_tools = CalendarToolkit().get_tools()

master_tools = gmail_tools + calendar_tools

print(f"Tools Ready: {len(master_tools)} loaded.")

# Bind tools to the model
llm_with_tools = llm.bind_tools(master_tools)

# Define the graph node
def agent_node(state: MessagesState):
    """Decides next step based on history."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Prebuilt tool node
tool_node = ToolNode(master_tools)

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

print("\n" + "="*40)
print("PlanX v1.0 is online")
print("(Type 'exit' or 'quit' to stop)")
print("="*40 + "\n")


# Initialize Chat Memory with System Prompt
current_state = {
    "messages": [SystemMessage(content=system_prompt)]
}

while True:
    try:
        # 1. Get User Input
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("PlanX shutting down. Goodbye!")
            break
        
        # 2. Add User Message to State
        current_state["messages"].append(HumanMessage(content=user_input))

        # 3. Run the Graph
        # We pass the accumulated state to the graph
        # invoke() returns the FINAL state after all tools have run
        final_state = app.invoke(current_state)

        # 4. Extract and Print Response
        # The last message in 'final_state' is the AI's final answer
        ai_response = final_state["messages"][-1]
        print(f"\nPlanX: {ai_response.content}\n")

        # 5. Update Memory
        # We save the final state so the next loop remembers what happened
        current_state = final_state

    except Exception as e:
        print(f"Error: {e}")
