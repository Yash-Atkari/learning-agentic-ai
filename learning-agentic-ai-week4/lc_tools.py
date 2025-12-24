import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# Setup the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    print(f"Tool called: Multiplying {a} * {b}")
    return a * b

# Bind the tool to the model
# This tells the LLM "You have permission to use this"
llm_with_tools = llm.bind_tools([multiply])

# 4. The agent part
def run_agent(user_prompt):
    # Ask the AI
    msg = llm_with_tools.invoke(user_prompt)
    
    # Check if AI wants to use a tool
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            print(f"AI is using tool: {tool_call['name']}")
            
            # Execute the actual Python function
            result = multiply.invoke(tool_call["args"])
            
            # Give the result back to AI
            final_resp = llm_with_tools.invoke([
                HumanMessage(content=user_prompt),
                msg,
                ToolMessage(tool_call_id=tool_call["id"], content=str(result))
            ])
            return final_resp.content
    else:
        return msg.content

# Test it
print(f"Result: {run_agent('What is 123 times 456?')}")
