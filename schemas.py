from pydantic import BaseModel, Field

# Define the "Mold" (Schema)

class PizzaOrder(BaseModel):
    item: str = Field(description="The name of the food item")
    flavor: str = Field(description="The variety or topping")
    quantity: int = Field(description="How many pizzas", default=1)

# Test it with GOOD data (Stimulating a perfect AI responses)

try:
    # AI response
    incoming_data = {"item": "pizza", "flavor": "cheese", "quantity": 5}

    # Pass it into model
    order = PizzaOrder(**incoming_data)

    print("Validation successful!")
    print(f"Order confirmed: {order.quantity}x {order.flavor} {order.item}")

    # Proof it's a real int
    print(f"The quantity is type: {type(order.quantity)}")

except Exception as e:
    print(f"Validation failed: {e}")

try:
    bad_data = {"quantity": 1, "flavor": "pepperoni"}

    order = PizzaOrder(**bad_data)
    print("Validation successful!")
    # print(order.flavor)
except Exception as e:
    print(f"Safety Net Caught an Error: {e}")
