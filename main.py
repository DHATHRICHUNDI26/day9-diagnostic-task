
import os
from typing import Literal

from google import genai

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from google.genai import types, errors
load_dotenv()


# Define the expected JSON structure
class MaintenanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(description="Type of maintenance issue")
    priority: Literal["High", "Medium", "Low"]
    summary: str
    recommended_action: str


# Analyze the maintenance request using Gemini
def analyze_request(request: str) -> MaintenanceResponse:

    if not request.strip():
        raise ValueError("Maintenance request cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        api_version="v1beta"
    )
)

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"""
            Analyze the following maintenance request:

            {request}

            Identify its category and priority.
            Provide a short summary and recommended action.
            Priority must be High, Medium, or Low.
            """,
            config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema={
        "type": "OBJECT",
        "properties": {
            "category": {
                "type": "STRING",
                "description": "Type of maintenance issue"
            },
            "priority": {
                "type": "STRING",
                "enum": ["High", "Medium", "Low"]
            },
            "summary": {
                "type": "STRING"
            },
            "recommended_action": {
                "type": "STRING"
            }
        },
        "required": [
            "category",
            "priority",
            "summary",
            "recommended_action"
        ]
    }
        )
           
        )

        if not response.text:
            raise RuntimeError("Empty response received from Gemini.")

        result = MaintenanceResponse.model_validate_json(
            response.text
        )

        return result

    except ValidationError as error:
        raise RuntimeError(
            f"JSON validation failed: {error}"
        ) from error

    except errors.APIError as error:
        raise RuntimeError(
            f"Gemini API error: {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(
        f"Unexpected error while processing request: {error}"
    ) from error


# Main function
def main():

    request = input("Enter maintenance request: ")

    try:
        result = analyze_request(request)

        print("\nValidated JSON Response:")
        print(result.model_dump_json(indent=2))

    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
