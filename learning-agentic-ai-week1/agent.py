from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Define the Mold (Schema)
class Order(BaseModel):
    item: str = Field(description="The main food item")
    flavor: str = Field(description="The variety, topping or style")
    quantity: int = Field(description="The number of items ordered")
    price_estimate: float = Field(description="Estimate cost: $15 per pizza, $2 per drink")

# User Input
user_command = "I'm super hungry! Get me four large pepperoni pizza and a coke."

try:
    # API Call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_command,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Order
        )
    )

    # Validation
    my_order = Order.model_validate_json(response.text)

    # Use the Data
    print("Order Received!")
    print(f"Item: {my_order.item}")
    print(f"Qty: {my_order.quantity}")
    print(f"Cost: ${my_order.price_estimate}")

    # Prove it's a real data, not text
    total = my_order.price_estimate * 1.1
    print(f"Total with Tax: ${total:.2f}")

    # : -> "I am about to give you special formatting instructions."
    # .2 -> "Keep exactly 2 digits after the decimal point."
    # f -> "Treat this as a float (a number with decimals)."

except Exception as e:
    print(f"Error: {e}")
