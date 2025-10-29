#!/usr/bin/env python3
"""
Integration tests for the complete search pipeline.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from server.ollama_adapter import OllamaAdapter, SearchResult, FallbackAdapter
from scraper.playwright_scraper import ScrapedContent, HTTPFallbackScraper
from formatter.format_with_model import ContentFormatter, FormattedContent, FormattedSection
from extractor.extract_answer import AnswerExtractor, ExtractedAnswer


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def create_test_scraped_content(self) -> ScrapedContent:
        """Create test scraped content."""
        return ScrapedContent(
            url="https://docs.python.org/3/tutorial/",
            title="Python Tutorial - Python 3.11 Documentation",
            text_content="""
            Python Tutorial
            
            Python is an easy to learn, powerful programming language. 
            It has efficient high-level data structures and a simple but effective approach to object-oriented programming.
            
            Using the Python Interpreter
            
            The Python interpreter is usually installed as /usr/local/bin/python3.11 on those machines where it is available.
            """,
            html_content="<html><head><title>Python Tutorial</title></head><body><h1>Python Tutorial</h1><p>Python is easy to learn...</p></body></html>",
            meta_tags={"description": "Official Python tutorial"},
            og_tags={"og:title": "Python Tutorial"},
            links=["https://docs.python.org/3/", "https://python.org"],
            images=["https://docs.python.org/3/_static/py.png"],
            pdf_links=[],
            content_length=300,
            load_time=1.2,
            status_code=200
        )
    
    @pytest.mark.asyncio
    async def test_ollama_adapter_fallback(self):
        """Test Ollama adapter with fallback."""
        # Test fallback adapter when Ollama is not available
        adapter = FallbackAdapter()
        
        results = await adapter.search("python tutorial", max_results=3)
        
        assert len(results) == 1
        assert results[0].url.startswith("https://google.com/search")
        assert "python tutorial" in results[0].url
        assert results[0].confidence == 0.5
        
        await adapter.close()
    
    @pytest.mark.asyncio
    async def test_content_formatter_fallback(self):
        """Test content formatter with fallback (no model)."""
        adapter = FallbackAdapter()
        formatter = ContentFormatter(adapter)
        
        scraped = self.create_test_scraped_content()
        formatted = await formatter.format_content(scraped, use_model=False)
        
        assert formatted.url == scraped.url
        assert formatted.title == scraped.title
        assert len(formatted.sections) > 0
        assert formatted.sections[0].heading in ["Main Content", "Python Tutorial"]
        assert "Python is" in formatted.sections[0].text
        
        await formatter.close()
    
    @pytest.mark.asyncio
    async def test_answer_extractor_fallback(self):
        """Test answer extractor with fallback (no model)."""
        adapter = FallbackAdapter()
        extractor = AnswerExtractor(adapter)
        
        # Create formatted content
        formatted = FormattedContent(
            url="https://docs.python.org/3/tutorial/",
            title="Python Tutorial",
            metadata={},
            sections=[
                FormattedSection(
                    heading="What is Python?",
                    text="Python is an easy to learn, powerful programming language created by Guido van Rossum.",
                    length=80
                ),
                FormattedSection(
                    heading="Installation",
                    text="The Python interpreter is usually installed as /usr/local/bin/python3.11 on Unix systems.",
                    length=90
                )
            ],
            raw_html_snippet="<html>...</html>",
            summary="Python programming tutorial",
            content_length=170
        )
        
        answer = await extractor.extract_answer("Who created Python?", formatted, use_model=False)
        
        assert answer.confidence > 0
        assert "guido" in answer.answer.lower() or "van rossum" in answer.answer.lower()
        assert answer.source_url == formatted.url
        
        await extractor.close()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_simulation(self):
        """Test complete pipeline with simulated components."""
        print("Running complete pipeline simulation...")
        
        # Step 1: Simulate URL search
        search_results = [
            SearchResult(
                url="https://docs.python.org/3/tutorial/",
                title="Python Tutorial",
                confidence=0.9,
                reason="Official Python documentation"
            ),
            SearchResult(
                url="https://python.org",
                title="Python Programming Language",
                confidence=0.8,
                reason="Official Python website"
            )
        ]
        
        print(f"✓ Step 1: Found {len(search_results)} candidate URLs")
        
        # Step 2: Simulate scraping
        scraped_contents = [
            self.create_test_scraped_content(),
            ScrapedContent(
                url="https://python.org",
                title="Python Programming Language",
                text_content="Python is a programming language that lets you work quickly and integrate systems more effectively.",
                html_content="<html><body>Python programming language</body></html>",
                meta_tags={},
                og_tags={},
                links=[],
                images=[],
                pdf_links=[],
                content_length=100,
                load_time=0.8,
                status_code=200
            )
        ]
        
        print(f"✓ Step 2: Scraped {len(scraped_contents)} pages")
        
        # Step 3: Format content
        adapter = FallbackAdapter()
        formatter = ContentFormatter(adapter)
        
        formatted_contents = await formatter.format_multiple(scraped_contents, use_model=False)
        
        assert len(formatted_contents) == 2
        assert all(len(fc.sections) > 0 for fc in formatted_contents)
        
        print(f"✓ Step 3: Formatted {len(formatted_contents)} pages")
        
        # Step 4: Extract answers
        extractor = AnswerExtractor(adapter)
        
        query = "What is Python?"
        extracted_answers = await extractor.extract_from_multiple(
            query, formatted_contents, use_model=False
        )
        
        assert len(extracted_answers) == 2
        assert all(ea.confidence > 0 for ea in extracted_answers)
        
        print(f"✓ Step 4: Extracted {len(extracted_answers)} answers")
        
        # Step 5: Combine answers
        best_answer = extractor.combine_answers(extracted_answers)
        
        assert best_answer is not None
        assert "python" in best_answer.answer.lower()
        assert best_answer.confidence > 0
        
        print(f"✓ Step 5: Best answer confidence: {best_answer.confidence:.2f}")
        print(f"   Answer: {best_answer.answer[:100]}...")
        
        # Cleanup
        await formatter.close()
        await extractor.close()
        
        print("✓ Complete pipeline simulation successful!")
        
        return {
            "query": query,
            "candidate_urls": len(search_results),
            "scraped_pages": len(scraped_contents),
            "formatted_pages": len(formatted_contents),
            "extracted_answers": len(extracted_answers),
            "best_answer_confidence": best_answer.confidence,
            "best_answer": best_answer.answer
        }


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling throughout the pipeline."""
    print("Testing error handling...")
    
    # Test with empty/invalid data
    adapter = FallbackAdapter()
    formatter = ContentFormatter(adapter)
    extractor = AnswerExtractor(adapter)
    
    # Test formatter with empty content
    empty_scraped = ScrapedContent(
        url="https://example.com",
        title="",
        text_content="",
        html_content="",
        meta_tags={},
        og_tags={},
        links=[],
        images=[],
        pdf_links=[],
        content_length=0,
        load_time=0,
        status_code=404,
        error="Page not found"
    )
    
    formatted = await formatter.format_content(empty_scraped, use_model=False)
    assert formatted.format_error is not None
    assert formatted.content_length == 0
    
    # Test extractor with empty formatted content
    empty_formatted = FormattedContent(
        url="https://example.com",
        title="Empty",
        metadata={},
        sections=[],
        raw_html_snippet="",
        summary="",
        content_length=0
    )
    
    answer = await extractor.extract_answer("test query", empty_formatted, use_model=False)
    assert answer.confidence == 0.0
    assert "No content available" in answer.answer
    
    await formatter.close()
    await extractor.close()
    
    print("✓ Error handling tests passed")


@pytest.mark.asyncio
async def test_performance_benchmarks():
    """Test performance of the pipeline components."""
    import time
    
    print("Running performance benchmarks...")
    
    adapter = FallbackAdapter()
    formatter = ContentFormatter(adapter)
    extractor = AnswerExtractor(adapter)
    
    # Create test data
    test_scraped = []
    for i in range(10):
        content = ScrapedContent(
            url=f"https://example{i}.com",
            title=f"Test Page {i}",
            text_content=f"This is test content for page {i}. " * 20,  # ~500 chars
            html_content=f"<html><body>Test {i}</body></html>",
            meta_tags={},
            og_tags={},
            links=[],
            images=[],
            pdf_links=[],
            content_length=500,
            load_time=1.0,
            status_code=200
        )
        test_scraped.append(content)
    
    # Benchmark formatting
    start_time = time.time()
    formatted_contents = await formatter.format_multiple(test_scraped, use_model=False)
    format_time = time.time() - start_time
    
    # Benchmark extraction
    start_time = time.time()
    extracted_answers = await extractor.extract_from_multiple(
        "test query", formatted_contents, use_model=False
    )
    extract_time = time.time() - start_time
    
    print(f"✓ Formatted {len(formatted_contents)} pages in {format_time:.2f}s ({format_time/len(formatted_contents):.3f}s per page)")
    print(f"✓ Extracted {len(extracted_answers)} answers in {extract_time:.2f}s ({extract_time/len(extracted_answers):.3f}s per answer)")
    
    # Performance assertions (reasonable thresholds for heuristic processing)
    assert format_time < 5.0  # Should format 10 pages in under 5 seconds
    assert extract_time < 5.0  # Should extract 10 answers in under 5 seconds
    
    await formatter.close()
    await extractor.close()
    
    print("✓ Performance benchmarks passed")


if __name__ == "__main__":
    print("Running integration tests...")
    
    # Run async tests
    async def run_all_tests():
        test_integration = TestIntegration()
        
        await test_integration.test_complete_pipeline_simulation()
        await test_error_handling()
        await test_performance_benchmarks()
        
        print("✓ All integration tests passed!")
    
    asyncio.run(run_all_tests())