from google import genai
import os
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Define the Mold (Schema)
class Ticket(BaseModel):
    category: str = Field(description="The urgency level: 'Urgent' or 'General'")
    department: str = Field(description="The team needed: 'DevOps', 'Support' or 'Billing'")
    reasoning: str = Field(description="The short explanation about your decision")

# User Input
user_command = "I'm very hungry! Please help me."

try:
    # API Call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_command,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Ticket
        )
    )

    # Validation
    my_ticket = Ticket.model_validate_json(response.text)

    print("Ticket Generated!")
    print(f"Ticket category: {my_ticket.category}")
    print(f"Ticket department: {my_ticket.department}")
    print(f"Reason: {my_ticket.reasoning}")

except Exception as e:
    print(f"Error: {e}")
