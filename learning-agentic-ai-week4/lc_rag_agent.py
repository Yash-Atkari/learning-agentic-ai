import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# Setup the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Setup embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Load vector database
print("Loading knowledge base...")
try:
    vectorstore = FAISS.load_local(
        "faiss_index_react", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Knowledge base loaded!")
except Exception as e:
    print("Could not load database.")
    exit()

# Define tools
@tool
def search_pdf(query: str) -> str:
    """Searches the pdf for the answer to the query."""
    print(f"Searching PDF for: {query}...")

    # Search the vector DB for the 3 most relevant chunks
    docs = vectorstore.similarity_search(query, k=3)

    # Combine the text from those chunks
    context = "\n".join([d.page_content for d in docs])
    return context

@tool
def multiply(a: int, b: int) -> int:
    """Multiplying two integers and returns the result"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two integers and returns the result"""
    return a + b

tools_map = {
    "search_pdf": search_pdf,
    "multiply": multiply,
    "add": add
}

llm_with_tools = llm.bind_tools([search_pdf, multiply, add])

# Memory
chat_history =[
    ("system", "You are Jarvis. Use 'search_pdf' to answer questions about documents. Use math tools for calculations.")
]

# The agent
def jarvis(user_prompt):
    print(f"User: {user_prompt}")
    # Add user message to memory
    chat_history.append(HumanMessage(content=user_prompt))

    # Call model with full history
    ai_msg = llm_with_tools.invoke(chat_history)
    chat_history.append(ai_msg)

    if ai_msg.tool_calls:
        print(f"Jarvis: (Thinking...) I need to use {len(ai_msg.tool_calls)} tool(s).")

        for tool_call in ai_msg.tool_calls:# Execute the correct tool
            tool_name = tool_call["name"]

            # Smart Lookup: No more if/elif!
            if tool_name in tools_map:
                selected_tool = tools_map[tool_name]
                print(f"Executing: {tool_name} with {tool_call['args']}")
                
                # Run it
                result = selected_tool.invoke(tool_call["args"])

                # E. Add tool result to memory
                chat_history.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))

        # Call model again using tool results included
        final_res = llm_with_tools.invoke(chat_history)
        # Add final answer to memory
        chat_history.append(final_res)

        print(f"Jarvis: {final_res.content}")
        return final_res.content
    else:
        print(f"Jarvis: {ai_msg.content}")
        return ai_msg.content
    
# 1. Tell it your name
# jarvis("Hi, I am Yash.")

# 2. Ask for Math (It uses tools)
# jarvis("What is 55 times 10?")

# 3. Ask for Name (It uses Memory)
# jarvis("What is my name? And add 5 to the previous math result.")

jarvis("What is the main topic of the document I uploaded?")
