from google import genai
import os
from google.genai import types
from dotenv import load_dotenv
from tools import get_weather

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Map: Connect string names to real functions
# AI sends the string 'get_weather'. We need to know which function it is.
function_map = {
    "get_weather": get_weather,
}

user_query = "What is the weather like today in New York city?"
print("User Query:", user_query)

try:
    response1 = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_query,

        config=types.GenerateContentConfig(
            # We pass the actual Python function here. 
            # The SDK automatically reads your docstring!
            tools=[get_weather],
            # This forces the model to PAUSE and give you the tool call request
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    print("\nAI Response:")

    # Check if the AI wants to call a function
    if response1.function_calls:
        for call in response1.function_calls:
            print(f"\nAI wants to call a function: {call.name} with {call.args}")

            # Lookup the real function from our map
            tool_function = function_map.get(call.name)

            if tool_function:
                # Execute the function. We unpack the dictionary arguments into the function.
                tool_result = tool_function(**call.args)

                print(f"Tool result: {tool_result}")

                # Send the result back to AI
                # We create a new history containing the result
                response2 = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        # We send the *entire* history back
                        # The user's question
                        types.Content(role="user", parts=[types.Part(text=user_query)]),

                        # The AI tool call request
                        response1.candidates[0].content,

                        # The result of the tool
                        types.Content(role="tool", parts=[types.Part(
                           function_response=types.FunctionResponse(
                                name=call.name,
                                response=tool_result
                           )
                        )])
                    ],
                    config=types.GenerateContentConfig(
                        tools=[get_weather]
                    )
                )

                print(f"\nFinal AI Response with Tool Result: {response2.text}")
    else:
        print(response1.text)

except Exception as e:
    print(f"Error: {e}")
