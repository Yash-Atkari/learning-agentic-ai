import time
import json
from google import genai
import os
from google.genai import types
from dotenv import load_dotenv

from tools import get_weather

load_dotenv()

my_persona = """
You are 'Orbit', a futuristic AI assistant for space travelers.
1. You always speak using terminology (e.g. "Copy that", "Trajectory calculated", "Affirmative").
2. You never say yes you say "Thrusters engaged".
3. When you use a tool, you annouce it as "Deploying subroutine".
"""

chat_file = "mission_log.json" # The file we save to

function_map = {
    "get_weather": get_weather
}

class SmartAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash"

        self.history = self.load_memory()

        # Define the tools we want to use
        self.tools_config = [get_weather, self.request_google_search]

    def request_google_search(self, query: str):
        """Dummy tool to trigger real google search"""

        return "placeholder"
    
    def load_memory(self):
        if os.path.exists(chat_file):
            print("Mission log found. Loading previous mission data...")

            try:
                with open(chat_file, "r") as f:
                    data = json.load(f) 
                    # We have to convert the JSON back into Google's 'types.Content' objects
                    restored_history = []
                    for item in data:
                        # Rebuild the parts list
                        parts = [types.Part(text=p['text']) for p in item['parts'] if 'text' in p]
                        # Note: This simple loader only handles text.
                        # Complex tool_calls are harder to save/load manually (Week 4 frameworks handle this).
                        if parts:
                            restored_history.append(types.Content(role=item['role'], parts=parts))
                    return restored_history
            except Exception as e:
                print(f"Corrupt log file. Starting fresh. Error: {e}")
        else:
            print("No previous logs. Initializing new mission.")
            return []
        
    def save_memory(self):
        # We convert complex Google objects into simple JSON text we can save
        serializable_history = []
        for item in self.history:
            # Only saving text parts for simplicity in this week's lesson
            text_parts = []
            for part in item.parts:
                if part.text:
                    text_parts.append({'text': part.text})

            if text_parts:
                serializable_history.append({
                    "role": item.role,
                    "parts": text_parts
                })

        with open(chat_file, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def chat(self, user_query: str):
        print(f"\nUser: {user_query}")

        # Add user to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_query)])
        )

        try:
            # We inject 'system_instruction' into the config
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    tools=self.tools_config,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    system_instruction=my_persona
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

                        # Get final answer based on tool result - We must send the system instruction again
                        final_res = self.client.models.generate_content(
                            model=self.model,
                            contents=self.history,
                            config=types.GenerateContentConfig(
                                tools=self.tools_config,
                                system_instruction=my_persona
                            )
                        )

                        final_output = final_res.text
            else:
                final_output = response.text

            # Save the final answer to memory
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=final_output)])
            )

            self.save_memory()

            return final_output

        except Exception as e:
            return f"Error: {e}"
        
if __name__ == "__main__":
    bot = SmartAgent()
    
    # print(f"Bot: {bot.chat('My name is Yash.')}")
    
    print(f"Bot: {bot.chat('What is my name?')}")
    