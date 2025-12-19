from google import genai
import os
from google.genai import types
from dotenv import load_dotenv

from tools import get_weather

load_dotenv()

function_map = {
    "get_weather": get_weather
}

class SmartAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash"

        self.history = []

        # Define the tools we want to use
        self.tools_config = [get_weather, self.request_google_search]

    def request_google_search(self, query: str):
        """Dummy tool to trigger real google search"""

        return "placeholder"
    
    def chat(self, user_query: str):
        print(f"User: {user_query}")

        # Add user to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_query)])
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    tools=self.tools_config,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )

            if response.function_calls:
                for call in response.function_calls:
                    print(f"Agent intent: calling {call.name} with {call.args}")

                    tool_function = function_map.get(call.name)

                    if call.name == "request_google_search":
                        search_query = call.args.get("query")
                        print(f"Running google search for {search_query}")

                        google_res = self.client.models.generate_content(
                            model=self.model,
                            contents=f"Answer this using google search: {user_query}",
                            config=types.GenerateContentConfig(
                                tools=[types.Tool(google_search=types.GoogleSearch())]
                            )
                        )

                        final_output = f"(Via Google Search): {google_res.text}"
                    else:
                        tool_res = tool_function(**call.args)
                        print(f"Tool result: {tool_res}")

                        # Add 'Model Call' to history
                        self.history.append(response.candidates[0].content)

                        # Add 'Tool Response' to history
                        self.history.append(
                            types.Content(
                                role="tool",
                                parts=[types.Part(
                                    function_response={
                                        "name": call.name,
                                        "response": tool_res
                                    }
                                )]
                            )
                        )

                        # Get final answer based on tool result
                        final_res = self.client.models.generate_content(
                            model=self.model,
                            contents=self.history,
                            config=types.GenerateContentConfig(
                                tools=self.tools_config
                            )
                        )

                        final_output = final_res.text
            else:
                final_output = response.text

            # Save the final answer to memory
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=final_output)])
            )

            return final_output

        except Exception as e:
            return f"Error: {e}"
        
if __name__ == "__main__":
    bot = SmartAgent()

    # Turn 1: Memory Test
    print(f"Bot: {bot.chat('Hi, I am Yash.')}")
    
    # Turn 2: Local Tool Test
    print(f"Bot: {bot.chat('What is the weather in Mumbai?')}")
    
    # Turn 3: Context Test (Does it remember where I checked weather?)
    print(f"Bot: {bot.chat('Is that city in India?')}")
    
    # Turn 4: Hybrid Search Test
    print(f"Bot: {bot.chat('Who is the CEO of OpenAI?')}")
