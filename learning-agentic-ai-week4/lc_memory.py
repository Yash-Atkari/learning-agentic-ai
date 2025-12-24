# 1. IMPORTS
import os  # To access operating system environment variables
from dotenv import load_dotenv  # To load the .env file with your passwords
# Import the specific Google Gemini wrapper for LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
# Import tools to build the "script" sent to the AI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import the "Manager" that handles memory automatically
from langchain_core.runnables.history import RunnableWithMessageHistory
# Import the specific type of memory storage (RAM memory)
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load the secret keys from .env file into the environment
load_dotenv()

# 2. MODEL SETUP
# Initialize the Google Gemini Model using LangChain's wrapper
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # Specify which Gemini version to use
    api_key=os.environ.get("GEMINI_API_KEY") # Grab the key from the environment
)

# 3. MEMORY STORAGE (The File Cabinet)
# Create an empty dictionary to store chat histories for ALL users
store = {}

# Define a function ("The Librarian") to find or create a specific user's history
def get_session_history(session_id: str):
    # If this user (session_id) is new and not in the dictionary...
    if session_id not in store:
        # ...create a fresh, empty memory list for them
        store[session_id] = InMemoryChatMessageHistory()
    # Return their specific history object
    return store[session_id]

# 4. PROMPT TEMPLATE (The Script)
# Define the structure of the message sent to the AI
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named Jarvis."), # System instruction (Persona)
    MessagesPlaceholder(variable_name="history"), # THE EMPTY CHAIR: Old messages get injected here
    ("human", "{input}"), # The slot where the user's NEW message goes
])

# 5. CREATE THE CHAIN (The Pipe)
# Connect the Prompt directly to the Model using the '|' pipe operator
# Data flows from Prompt -> Model
chain = prompt | model

# 6. ADD MEMORY MANAGEMENT (The Manager)
# Wrap the basic chain with the "Memory Manager"
conversation_chain = RunnableWithMessageHistory(
    chain,                       # The basic chain to wrap
    get_session_history,         # The function to find the user's history
    input_messages_key="input",  # Tell it: "The user's new text is in the 'input' variable"
    history_messages_key="history" # Tell it: "Put old messages in the 'history' placeholder"
)

# 7. RUNNING IT (Session 123)
# Invoke (Run) the chain for a specific user ID
response1 = conversation_chain.invoke(
    {"input": "Hi, I am Yash."},              # The user's message
    config={"configurable": {"session_id": "123"}} # The ID of the user (The key for the dictionary)
)
# Print the result
print(f"User: Hi, I am Yash. \nBot: {response1.content}\n")

# Run it again for the SAME user (Session 123)
# Because the ID is "123", it will find the previous "Hi I am Yash" in the store
response2 = conversation_chain.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "123"}}
)
print(f"User: What is your name? \nBot: {response2.content}")
