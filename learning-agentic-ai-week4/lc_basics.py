import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Setup the model
# We use langchain's wrapper for Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Create a prompt template
# This replaces f-strings. It's cleaner and safer.
prompt = ChatPromptTemplate.from_template(
    "You are a comedian. Tell me a short joke about {topic}."
)

# Create the chain
# The pipe operator '|' connects the prompt to the model automatically.
chain = prompt | llm

# Run it
# We just pass the variables (topic) and langchain handles the rest.
response = chain.invoke({"topic": "girls"})

print(response.content)
