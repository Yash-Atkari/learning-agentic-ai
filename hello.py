# Import Google GenAI library to use the AI models
from google import genai
from google.genai import types
import json
# Import 'os' to access the computer's env variables
import os

# Import the function to load .env file
from dotenv import load_dotenv

# Searches for a .env file and loads the variables into python
load_dotenv()

# Create the client. It grabs the key securely from the loaded env
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

prompt = """
You are a pizza ordering assistant. 
Extract the order details from the user's request.
Return the result in this JSON format:
{
    "item": "name of item",
    "flavor": "flavor or type",
    "size": "size if mentioned, else medium"
}

User Request: "I want a huge pepperoni pizza"
"""

try:
    # Sends a message to the AI model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    # Print the success message
    print("Success! AI says:")
    # Print the actual text response from the AI
    print(response.text)
    # print(response.text["item"])
    # print(response.text[10])
    data = json.loads(response.text)
    print(data)
    print(data["item"])
    print(data["flavor"])
    print(data["sizes"])
except Exception as e:
    # If sometihng breaks, print the error so we can fix it
    print(f"Error: {e}")
    