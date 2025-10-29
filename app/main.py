#!/usr/bin/env python3
"""
Main FastAPI application for the Search Query Agent.
Provides /search endpoint for personalized search with scraping and extraction.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from server.ollama_adapter import OllamaAdapter, create_adapter, SearchResult
from scraper.playwright_scraper import PlaywrightScraper, create_scraper
from formatter.format_with_model import ContentFormatter
from extractor.extract_answer import AnswerExtractor


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    max_urls: int = Field(default=5, description="Maximum URLs to scrape", ge=1, le=10)
    use_model_formatting: bool = Field(default=True, description="Use model for content formatting")
    use_model_extraction: bool = Field(default=True, description="Use model for answer extraction")


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    answer: str
    confidence: float
    reasoning: str
    sources: List[Dict[str, Any]]
    processing_time: float
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    model_ready: bool
    timestamp: float


# Global components (initialized in lifespan)
app_state = {
    "adapter": None,
    "scraper": None,
    "formatter": None,
    "extractor": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    print("Starting Search Query Agent...")
    
    try:
        # Initialize components
        print("Initializing Ollama adapter...")
        app_state["adapter"] = create_adapter()
        
        print("Initializing web scraper...")
        app_state["scraper"] = await create_scraper(use_playwright=True, headless=True)
        
        print("Initializing content formatter...")
        app_state["formatter"] = ContentFormatter(app_state["adapter"])
        
        print("Initializing answer extractor...")
        app_state["extractor"] = AnswerExtractor(app_state["adapter"])
        
        print("✓ All components initialized successfully")
        
        yield
        
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        # Continue with limited functionality
        yield
    
    finally:
        # Shutdown
        print("Shutting down Search Query Agent...")
        
        if app_state["adapter"]:
            await app_state["adapter"].close()
        
        if app_state["scraper"]:
            await app_state["scraper"].stop()
        
        if app_state["formatter"]:
            await app_state["formatter"].close()
        
        if app_state["extractor"]:
            await app_state["extractor"].close()
        
        print("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Search Query Agent",
    description="Personalized search agent with QLoRA fine-tuning and content extraction",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Search Query Agent API",
        "version": "0.1.0",
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check Ollama availability
        ollama_available = False
        model_ready = False
        
        if app_state["adapter"]:
            ollama_available = await app_state["adapter"].check_ollama_status()
            model_ready = ollama_available  # Simplified check
        
        return HealthResponse(
            status="healthy" if model_ready else "degraded",
            ollama_available=ollama_available,
            model_ready=model_ready,
            timestamp=time.time()
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            ollama_available=False,
            model_ready=False,
            timestamp=time.time()
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Main search endpoint."""
    start_time = time.time()
    
    try:
        # Validate components
        if not app_state["adapter"]:
            raise HTTPException(status_code=503, detail="Ollama adapter not available")
        
        if not app_state["scraper"]:
            raise HTTPException(status_code=503, detail="Web scraper not available")
        
        # Step 1: Get candidate URLs from fine-tuned model
        print(f"Searching for: {request.query}")
        search_results = await app_state["adapter"].search(request.query, request.max_urls)
        
        if not search_results:
            return SearchResponse(
                query=request.query,
                answer="No relevant URLs found in your browsing history.",
                confidence=0.0,
                reasoning="The personalized model did not find any matching URLs",
                sources=[],
                processing_time=time.time() - start_time,
                status="no_results"
            )
        
        print(f"Found {len(search_results)} candidate URLs")
        
        # Step 2: Scrape the top URLs
        urls_to_scrape = [result.url for result in search_results]
        scraped_contents = await app_state["scraper"].scrape_urls(urls_to_scrape)
        
        # Filter out failed scrapes
        successful_scrapes = [s for s in scraped_contents if not s.error and s.text_content]
        
        if not successful_scrapes:
            return SearchResponse(
                query=request.query,
                answer="Found relevant URLs but could not access their content.",
                confidence=0.2,
                reasoning="URLs were identified but scraping failed",
                sources=[{"url": r.url, "title": r.title, "confidence": r.confidence} 
                        for r in search_results],
                processing_time=time.time() - start_time,
                status="scraping_failed"
            )
        
        print(f"Successfully scraped {len(successful_scrapes)} pages")
        
        # Step 3: Format scraped content
        formatted_contents = await app_state["formatter"].format_multiple(
            successful_scrapes, 
            use_model=request.use_model_formatting
        )
        
        # Step 4: Extract answers
        extracted_answers = await app_state["extractor"].extract_from_multiple(
            request.query,
            formatted_contents,
            use_model=request.use_model_extraction
        )
        
        # Step 5: Combine and rank answers
        best_answer = app_state["extractor"].combine_answers(extracted_answers)
        
        if not best_answer:
            return SearchResponse(
                query=request.query,
                answer="Could not extract a specific answer from the content.",
                confidence=0.1,
                reasoning="Content was found but no specific answer could be extracted",
                sources=[{"url": s.url, "title": s.title} for s in successful_scrapes],
                processing_time=time.time() - start_time,
                status="extraction_failed"
            )
        
        # Step 6: Prepare response
        sources = []
        for i, (answer, scraped, formatted) in enumerate(
            zip(extracted_answers, successful_scrapes, formatted_contents)
        ):
            sources.append({
                "url": scraped.url,
                "title": scraped.title,
                "confidence": answer.confidence,
                "summary": formatted.summary,
                "sections_count": len(formatted.sections),
                "content_length": scraped.content_length,
                "load_time": scraped.load_time
            })
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            answer=best_answer.answer,
            confidence=best_answer.confidence,
            reasoning=best_answer.reasoning,
            sources=sources,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        return SearchResponse(
            query=request.query,
            answer=f"Search failed due to an error: {str(e)}",
            confidence=0.0,
            reasoning="Internal server error during search processing",
            sources=[],
            processing_time=time.time() - start_time,
            status="error"
        )


@app.get("/search/test")
async def test_search():
    """Test endpoint with a simple query."""
    test_request = SearchRequest(
        query="python tutorial",
        max_urls=3,
        use_model_formatting=False,  # Use heuristics for testing
        use_model_extraction=False
    )
    
    return await search(test_request, BackgroundTasks())


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 handler."""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/", "/search", "/health", "/docs"]}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500 handler."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )


def main():
    """Run the FastAPI application."""
    import os
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"Starting Search Query Agent API on {host}:{port}")
    print("Available endpoints:")
    print(f"  - API: http://{host}:{port}")
    print(f"  - Docs: http://{host}:{port}/docs")
    print(f"  - Health: http://{host}:{port}/health")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production
        access_log=True
    )


if __name__ == "__main__":
    main()
