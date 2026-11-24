
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Initialize client using v1beta API
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        api_version="v1beta"
    )
)

try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say hello in one sentence."
    )

    print("Gemini API Connection Successful!")
    print("\nGemini Response:")
    print(response.text)

except Exception as error:
    print("Gemini API Error:", error)
