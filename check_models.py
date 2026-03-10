import os
from dotenv import load_dotenv
from google import genai

load_dotenv() # Load environment variables from a .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # Initialize the Google GenAI client using the API key from environment variables
# Ensure GEMINI_API_KEY is correctly set in your .env file

print("--- Printing ALL available models ---")
try: # Retrieve the list of all models accessible to the current API key
    models = client.models.list()
    for m in models: # Each model object contains its unique name and the actions it can perform
        print(f"Name: {m.name} | Actions: {m.supported_actions}")
except Exception as e: # Graceful error handling for network issues or invalid API keys
    print(f"API Error: {e}")