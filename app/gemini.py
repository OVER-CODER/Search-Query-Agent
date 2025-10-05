import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

FUNCTION_SCHEMA = {
    "name": "get_candidate_urls",
    "description": "Return structured list of candidate URLs for the user query.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                        "required": ["url"],
                },
            },
        },
        "required": ["urls"],
    },
}


def ask_gemini_for_urls(user_query: str, top_k: int = 5) -> List[Dict]:
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        f"You are a helpful web research assistant. "
        f"Find the top {top_k} websites where the answer to the following query can be found. "
        f"Return your result via the 'get_candidate_urls' function."
        f"\n\nQuery: {user_query}"
    )

    response = model.generate_content(
        contents=[{"role": "user", "parts": [prompt]}],
        tools=[{"function_declarations": [FUNCTION_SCHEMA]}],
        tool_config={"function_calling_config": {"mode": "ANY"}}
    )

    urls = []
    try:
        part = response.candidates[0].content.parts[0]
        if hasattr(part, "function_call") and part.function_call:
            args_map = part.function_call.args
            urls = list(args_map.get("urls", []))
        else:
            raise ValueError("No valid function_call found in response.")
    except Exception as e:
        print("Raw Gemini response:\n", response)
        raise RuntimeError(f"Failed to parse Gemini output: {e}")

    return urls[:top_k]
