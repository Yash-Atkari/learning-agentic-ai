# Failed!
import wikipedia

def search_web(query: str):
    """
    Searches Wikipedia for factual information.
    Use this when you need to know about events, history, or people.
    
    Args:
        query (str): The search term (e.g. "IPL 2024 winner").
        
    Returns:
        str: A summary of the Wikipedia page.
    """
    print(f"Searching Wikipedia for: {query}")
    
    try:
        # 1. Search for the most relevant page
        # This returns a list of page titles, we take the first one
        search_results = wikipedia.search(query)
        
        if not search_results:
            return "No results found."
            
        first_result = search_results[0]
        
        # 2. Get the summary of that page
        # sentences=2 keeps it short for the AI
        summary = wikipedia.summary(first_result, sentences=3)
        
        return f"Page: {first_result}\nSummary: {summary}"
            
    except wikipedia.exceptions.DisambiguationError as e:
        # This happens if you search "Apple" (Fruit vs Company)
        return f"Ambiguous search. Options: {e.options[:5]}"
        
    except Exception as e:
        return f"Search error: {e}"

if __name__ == "__main__":
    # Test it
    print(search_web("Who won the last IPL 2024?"))
