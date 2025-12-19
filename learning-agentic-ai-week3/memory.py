from google import genai
import os
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ChatSession:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash"

        # The 'State': This list holds the entire conversation
        self.history = []

    def send_message(self, user_query):
        """
        Takes user input, adds it to history, send ALL history to AI, and stores the AI response.
        """

        # Add the user's message to our local history
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_query)]
            )
        )

        try:
            print(f"Sending {len(self.history)} messages to the brain...")

            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history
            )

            # Add the AI's response to our local history
            self.history.append(response.candidates[0].content)

            return response.text
        
        except Exception as e:
            return f"Error: {e}"
        
if __name__ == "__main__":
    # Create the branch/instance of class ChatSession
    bot = ChatSession()

    print("--Turn 1--")
    print(f"Bot: {bot.send_message("Hi, My name is Yash.")}")

    print("--Turn 2--")
    print(f"Bot: {bot.send_message("What is the capital of France?")}")

    print("--Turn 3--")
    print(f"Bot: {bot.send_message("What is my name?")}")
