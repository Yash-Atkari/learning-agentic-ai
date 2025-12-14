from google import genai
import os
from google.genai import types
from dotenv import load_dotenv

from tools import get_weather

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# We use this just to catch the AI's "Intent" to search.
def request_google_search(query: str):
    """
    Use this tool to search Google for current events or facts.
    """
    return "This is a placeholder."

function_map = {
    "get_weather": get_weather,
    "request_google_search": request_google_search # Map the dummy tool
}

def run_agent(user_query):
    try:
        print(f"User: {user_query}")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config=types.GenerateContentConfig(
                tools=[
                    # Custom tool
                    get_weather, 
                    
                    # Give fake tool to catch google search
                    request_google_search
                ],
                # We keep this disabled so we can handle get_weather manually.
                # If we use built-in google search it work automatically because it's server-side.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )

        # The AI wants to use tools
        if response.function_calls:
            for call in response.function_calls:
                print(f"Agent decided to use custom tool: {call.name}")
                
                tool_function = function_map.get(call.name)
                
                if call.name == "request_google_search":
                    print("Triggering real google search API...")

                    # Grab the query AI wanted to search for
                    search_query = call.args.get('query')

                    # Make a call specifically for Google Search
                    google_res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"Please answer this using Google Search: {search_query}",
                        config=types.GenerateContentConfig(
                            # Enable real built-in tool here
                            tools=[types.Tool(google_search=types.GoogleSearch())]
                        )
                    )

                    # Print the Google answer
                    print(f"Google Search result: {google_res.text}")
                else:
                    tool_result = tool_function(**call.args)
                    print(f"Tool result: {tool_result}")
                    
                    # Send result back
                    custom_tool_res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=user_query)]),
                            response.candidates[0].content,
                            types.Content(role="tool", parts=[types.Part(
                                function_response={
                                    "name": call.name,
                                    "response": tool_result
                                }
                            )])
                        ],
                        # Enable tools again for the final answer
                        config=types.GenerateContentConfig(
                            tools=[get_weather, request_google_search]
                        )
                    )
                    print(f"Custom Tool result: {custom_tool_res.text}")
        else:
            print(f"Assistant: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Search (Automatic Google Grounding)
    run_agent("Who won the last IPL?")

    # Weather (Manual Tool Loop)
    # run_agent("What is the weather in Tokyo?")
