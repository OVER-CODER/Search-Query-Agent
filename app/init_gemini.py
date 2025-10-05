import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")

genai.configure(api_key=API_KEY)


def ask_gemini_for_urls(user_query: str, top_k: int = 5) -> List[Dict]:
    """
    Calls Gemini to request top_k candidate URLs for the query.
    Returns list of dicts: [{'url':..., 'snippet':..., 'confidence':...}, ...]
    """
    system_prompt = (
        "You are a web-research assistant. Given a user search query, return a JSON array of "
        f"the top {top_k} URLs where the answer can be found. "
        "Respond strictly in valid JSON format: "
        "[{\"url\": \"...\", \"snippet\": \"...\", \"confidence\": 0.9}, ...]"
    )

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(
        f"{system_prompt}\n\nQuery: {user_query}"
    )

    text = response.text.strip()
    try:
        urls = json.loads(text)
        if not isinstance(urls, list):
            raise ValueError("Response is not a list")
    except Exception as e:
        print("Raw response:\n", text)
        raise e

    return urls[:top_k]


def main():
    query = input("Query: ")
    urls = ask_gemini_for_urls(query, top_k=5)
    print("\nTop URLs:")
    for u in urls:
        print("-", u["url"])


if __name__ == "__main__":
    main()
