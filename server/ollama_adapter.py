#!/usr/bin/env python3
"""
Ollama adapter for personalized search queries.
Handles communication with local Ollama instance.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
import httpx
import os
from dataclasses import dataclass
from ollama._types import ResponseError
import ollama


@dataclass
class SearchResult:
    """Structured search result."""
    url: str
    title: str
    confidence: float
    reason: str
    domain: Optional[str] = None


class OllamaAdapter:
    """Adapter for Ollama model inference."""
    
    def __init__(self, 
                 host: str = "http://localhost:11434",
                 model_name: str = "llama3.1:8b",
                 timeout: int = 30):
        self.host = host
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # System prompt for search queries
        self.system_prompt = """You are a personalized search assistant trained on the user's browsing history. 
When given a search query, return the most relevant URLs from their history as JSON.

Response format:
{
  "urls": [
    {
      "url": "https://example.com",
      "title": "Page Title", 
      "confidence": 0.95,
      "reason": "Explanation why this URL matches"
    }
  ]
}

Only return URLs that are highly relevant to the query. If no good matches exist, return an empty urls array.
Always respond with valid JSON only, no additional text."""
    
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = await self.client.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            return any(self.model_name in name for name in model_names)
            
        except Exception as e:
            print(f"Ollama status check failed: {e}")
            return False
    
    async def pull_model_if_needed(self) -> bool:
        """Pull model if not available."""
        try:
            if await self.check_ollama_status():
                return True
            
            print(f"Model {self.model_name} not found. Attempting to pull...")
            
            # Use synchronous client for model pulling
            try:
                ollama.pull(self.model_name)
                print(f"Successfully pulled {self.model_name}")
                return True
            except Exception as e:
                print(f"Failed to pull model: {e}")
                return False
                
        except Exception as e:
            print(f"Error checking/pulling model: {e}")
            return False
    
    def create_search_prompt(self, query: str) -> str:
        """Create prompt for search query."""
        return f"Given query: '{query}', which URL from my browsing history best answers it?"
    
    async def query_model(self, query: str) -> Dict[str, Any]:
        """Query the model for search results."""
        try:
            # Check if model is available
            if not await self.check_ollama_status():
                if not await self.pull_model_if_needed():
                    raise Exception(f"Model {self.model_name} not available and could not be pulled")
            
            # Create messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.create_search_prompt(query)}
            ]
            
            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = await self.client.post(
                f"{self.host}/api/chat",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            return result
            
        except Exception as e:
            print(f"Error querying model: {e}")
            # Return fallback response
            return {
                "message": {
                    "content": json.dumps({
                        "urls": [{
                            "url": "https://google.com/search?q=" + query.replace(" ", "+"),
                            "title": f"Search for: {query}",
                            "confidence": 0.3,
                            "reason": "Fallback search (model unavailable)"
                        }]
                    })
                }
            }
    
    def parse_response(self, response: Dict[str, Any]) -> List[SearchResult]:
        """Parse model response into SearchResult objects."""
        try:
            # Extract content from response
            content = response.get("message", {}).get("content", "")
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Extract URLs
            urls_data = data.get("urls", [])
            if not isinstance(urls_data, list):
                raise ValueError("URLs field must be a list")
            
            results = []
            for url_data in urls_data:
                if not isinstance(url_data, dict):
                    continue
                
                result = SearchResult(
                    url=url_data.get("url", ""),
                    title=url_data.get("title", ""),
                    confidence=float(url_data.get("confidence", 0.0)),
                    reason=url_data.get("reason", ""),
                    domain=url_data.get("domain")
                )
                
                # Validate required fields
                if result.url and result.confidence > 0:
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            
            # Return empty results on parse error
            return []
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Main search function."""
        if not query.strip():
            return []
        
        try:
            # Query model
            response = await self.query_model(query)
            
            # Parse results
            results = self.parse_response(response)
            
            # Sort by confidence and limit results
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class FallbackAdapter:
    """Fallback adapter when Ollama is not available."""
    
    def __init__(self):
        pass
    
    async def check_ollama_status(self) -> bool:
        return False
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Return fallback search results."""
        return [
            SearchResult(
                url=f"https://google.com/search?q={query.replace(' ', '+')}",
                title=f"Google Search: {query}",
                confidence=0.5,
                reason="Fallback search (Ollama not available)"
            )
        ]
    
    async def close(self):
        pass


def create_adapter(
    host: str = None,
    model_name: str = None,
    timeout: int = 30
) -> OllamaAdapter:
    """Factory function to create adapter with environment defaults."""
    
    # Get defaults from environment
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = model_name or os.getenv("MODEL_NAME", "llama3.1:8b")
    
    return OllamaAdapter(host=host, model_name=model_name, timeout=timeout)


async def main():
    """Test function."""
    adapter = create_adapter()
    
    try:
        # Test search
        results = await adapter.search("python tutorial")
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.url}")
            print(f"   Title: {result.title}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Reason: {result.reason}")
            print()
    
    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())