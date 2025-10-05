import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.gemini import ask_gemini_for_urls
from app.scrapper import scrape_urls_concurrent
from app.extractor import extract_best_snippet

app = FastAPI(title="LLM Search Agent")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
async def search(req: QueryRequest):
    query = req.query.strip()
    top_k = min(max(1, req.top_k), 10)

    try:
        candidates = ask_gemini_for_urls(query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failure: {str(e)}")

    urls = [c.get("url") for c in candidates if c.get("url")]
    if not urls:
        raise HTTPException(status_code=404, detail="No candidate URLs returned by LLM")

    scraped = await scrape_urls_concurrent(urls, concurrency=3)

    responses = []
    for s in scraped:
        snippet_info = extract_best_snippet(s.get("html", ""), query)
        responses.append({
            "url": s["url"],
            "title": s.get("title"),
            "status": s.get("status"),
            "snippet": snippet_info["snippet"],
            "score": snippet_info["score"]
        })

    ranked = sorted(responses, key=lambda x: x["score"], reverse=True)
    top_result = ranked[0] if ranked else None
    return {
        "query": query,
        "top_result": top_result,
        "candidates": ranked
    }

# To run: uvicorn app.main:app --reload --port 8000
