# This is a 'Mock' function.
def get_weather(city: str):
    # Docstring
    """
    Generate the current weather for a specific city.

    Args:
        city (str): The name of the city (e.g., "London", "New York").

    Returns:
        dict: A dictionary containing temperature and condition.
    """

    # Normalize input to handle case sensitivity
    city = city.lower().strip()

    if "london" in city:
        return {"temp": "15C", "condition": "Cloudy"}
    elif "tokyo" in city:
        return {"temp": "12C", "condition": "Rainy"}
    elif "san francisco" in city:
        return {"temp": "20C", "condition": "Sunny"}
    elif "nagpur" in city:
        return {"temp": "32C", "condition": "Hot"}
    else:
        # Default fallback for unknown cities
        return {"temp": "25C", "condition": "Clear"}
    
# Test it manually
if __name__ == "__main__":
    # This block only runs if you run 'python tools.py' directly
    print(get_weather("Nagpur"))
    print(get_weather("Tokyo"))
    print(get_weather("New York"))
