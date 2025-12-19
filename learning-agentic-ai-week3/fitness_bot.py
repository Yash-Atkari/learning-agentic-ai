import json
from google import genai
import os
from google.genai import types 
from dotenv import load_dotenv

from tools import calc_bmi

load_dotenv()

persona = """
You are a fitness coach. 
When user gives you his height and weight you must call calc_bmi. 
Based on the bmi you also tell the user that Is he/she healthy or not? 
If not healthy you have to give some dietry suggestions and workout plans. 
Your tone is simple, calm and like a supportive big brother. 
According to the need you can call request_google_search when there is a need of live events, facts and current info.
"""

chat_file = "fitness_log.json"

function_map = {
    "calc_bmi": calc_bmi
}

class Coach:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = "gemini-2.5-flash"

        self.tools_config = [
            self.request_google_search,
            calc_bmi
        ]

        self.history = self.load_memory()

    def request_google_search(self, query: str):
        """
        Dummy tool to trigger real google_search
        """

        return "placeholder"
    
    def save_memory(self):
        serializable_history = []

        for item in self.history:
            text_parts = []
            for part in item.parts:
                if part.text:
                    text_parts.append({'text': part.text})

            if text_parts:
                serializable_history.append({
                    'role': item.role,
                    'parts': text_parts
                })
        
        with open(chat_file, "w") as f:
            json.dump(serializable_history, f, indent=2)

    def load_memory(self):
        if os.path.exists(chat_file):
            print("Fitness log found. Loading previous fitness data...")

            try:
                with open(chat_file, "r") as f:
                    data = json.load(f)
                    restored_history = []
                    for item in data:
                        parts = [types.Part(text=p['text']) for p in item['parts'] if 'text' in p]

                        if parts:
                            restored_history.append(types.Content(role=item['role'], parts=parts))
                    return restored_history
            except Exception as e:
                print(f"Corrupt log file. Starting fresh. Error: {e}")
        else:
            print("No previous logs. Initialing fresh plan.")
            return []
    
    def chat(self, user_query: str):
        print(f"User: {user_query}")

        self.history.append(
            types.Content(role='user', parts=[types.Part(text=user_query)])
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    tools=self.tools_config,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    system_instruction=persona
                )
            )

            if response.function_calls:
                for call in response.function_calls:
                    print(f"AI itent: calling {call.name} with {call.args}")

                    tool_function = function_map.get(call.name)

                    if call.name == "request_google_search":
                        search_query = call.args.get('query')
                        print(f"Searching for {user_query}")

                        google_res = self.client.models.generate_content(
                            model=self.model,
                            contents=f"Answer this using google search {search_query}",
                            config=types.GenerateContentConfig(
                                tools=[types.Tool(google_search=types.GoogleSearch())]
                            )
                        )

                        final_output = f"(Via google search): {google_res.text}"
                    else:
                        tool_res = tool_function(**call.args)
                        print(f"Tool result: {tool_res}")

                        self.history.append(response.candidates[0].content)

                        self.history.append(
                            types.Content(
                                role='tool',
                                parts=[types.Part(
                                    function_response={
                                        "name": call.name,
                                        "response": {"result": tool_res}
                                    }
                                )]
                            )
                        )

                        final_res = self.client.models.generate_content(
                            model=self.model,
                            contents=self.history,
                            config=types.GenerateContentConfig(
                                tools=self.tools_config,
                                system_instruction=persona
                            )
                        )

                        final_output = final_res.text
            else:
                final_output = response.text

            self.history.append(
                types.Content(
                    role='model',
                    parts=[types.Part(text=final_output)]
                )
            )

            self.save_memory()

            return final_output
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    coach = Coach()

    # print(f"Coach: {coach.chat("Hi! My name is Yash.")}")

    # print(f"Coach: {coach.chat("My height is 5m and weight is 60kg.")}")

    # print(f"Coach: {coach.chat("Who is the prime minister of India?")}")

    print(f"Bot: {coach.chat("What is my name? Do you remember? Also tell my previous bmi count.")}")
