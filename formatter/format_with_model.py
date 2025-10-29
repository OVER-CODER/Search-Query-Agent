#!/usr/bin/env python3
"""
Content formatter using local model.
Converts scraped HTML into structured JSON.
"""

import json
import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from server.ollama_adapter import OllamaAdapter, create_adapter
from scraper.playwright_scraper import ScrapedContent


@dataclass
class FormattedSection:
    """Structured content section."""
    heading: str
    text: str
    length: int


@dataclass 
class FormattedContent:
    """Structured page content."""
    url: str
    title: str
    metadata: Dict[str, Any]
    sections: List[FormattedSection]
    raw_html_snippet: str
    summary: str
    content_length: int
    format_error: Optional[str] = None


class ContentFormatter:
    """Format scraped content using local model."""
    
    def __init__(self, adapter: OllamaAdapter = None):
        self.adapter = adapter or create_adapter()
        
        # System prompt for content formatting
        self.format_prompt = """You are a content formatting assistant. 
Given HTML content and metadata, extract and format it into structured JSON.

Response format:
{
  "title": "Page title",
  "summary": "Brief 2-3 sentence summary of the content",
  "sections": [
    {
      "heading": "Section heading or topic",
      "text": "Main content text",
      "length": 123
    }
  ],
  "metadata": {
    "og": {"title": "...", "description": "..."},
    "meta": {"description": "...", "keywords": "..."}
  }
}

Focus on the main content, ignore navigation, advertisements, and boilerplate text.
Respond with valid JSON only."""
    
    def clean_html_for_model(self, html_content: str, max_length: int = 4000) -> str:
        """Clean and truncate HTML for model processing."""
        # Remove script, style, and other non-content tags
        import re
        
        # Remove script and style blocks
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove comments
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        # Remove excess whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        
        # Truncate if too long
        if len(html_content) > max_length:
            html_content = html_content[:max_length] + "..."
        
        return html_content
    
    def extract_sections_from_text(self, text_content: str) -> List[FormattedSection]:
        """Extract sections from text content using heuristics."""
        sections = []
        
        # Split by potential headings (lines that look like headings)
        lines = text_content.split('\n')
        current_section = {"heading": "Main Content", "text": "", "length": 0}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a heading
            is_heading = (
                len(line) < 100 and  # Short lines
                (line.isupper() or  # ALL CAPS
                 line.istitle() or  # Title Case
                 line.endswith(':') or  # Ends with colon
                 (len(line.split()) <= 6 and not line.endswith('.')))  # Short, no period
            )
            
            if is_heading and current_section["text"]:
                # Save current section and start new one
                current_section["length"] = len(current_section["text"])
                sections.append(FormattedSection(**current_section))
                current_section = {"heading": line, "text": "", "length": 0}
            else:
                # Add to current section
                if current_section["text"]:
                    current_section["text"] += " "
                current_section["text"] += line
        
        # Add final section
        if current_section["text"]:
            current_section["length"] = len(current_section["text"])
            sections.append(FormattedSection(**current_section))
        
        # Merge very short sections
        merged_sections = []
        for section in sections:
            if section.length < 100 and merged_sections:
                # Merge with previous section
                prev = merged_sections[-1]
                prev.text += f" {section.heading}: {section.text}"
                prev.length = len(prev.text)
            else:
                merged_sections.append(section)
        
        return merged_sections[:10]  # Limit to 10 sections
    
    def create_fallback_format(self, scraped: ScrapedContent) -> FormattedContent:
        """Create formatted content without model (fallback)."""
        # Extract sections using heuristics
        sections = self.extract_sections_from_text(scraped.text_content)
        
        # Create metadata
        metadata = {
            "og": scraped.og_tags,
            "meta": scraped.meta_tags
        }
        
        # Create summary from first paragraph
        text_lines = scraped.text_content.split('\n')
        summary_lines = []
        for line in text_lines:
            line = line.strip()
            if line and len(line) > 50:  # Substantial content
                summary_lines.append(line)
                if len(' '.join(summary_lines)) > 200:
                    break
        
        summary = ' '.join(summary_lines)[:300] + "..." if summary_lines else "No summary available"
        
        # Get HTML snippet
        html_snippet = scraped.html_content[:500] + "..." if scraped.html_content else ""
        
        return FormattedContent(
            url=scraped.url,
            title=scraped.title or "Untitled",
            metadata=metadata,
            sections=sections,
            raw_html_snippet=html_snippet,
            summary=summary,
            content_length=scraped.content_length
        )
    
    async def format_with_model(self, scraped: ScrapedContent) -> FormattedContent:
        """Format content using the local model."""
        try:
            # Prepare content for model
            clean_html = self.clean_html_for_model(scraped.html_content)
            
            # Create prompt
            prompt = f"""Format this webpage content into structured JSON:

URL: {scraped.url}
Title: {scraped.title}
Meta description: {scraped.meta_tags.get('description', '')}

HTML Content:
{clean_html}

Text Content (first 1000 chars):
{scraped.text_content[:1000]}

Please format this into the required JSON structure."""
            
            # Query model using the chat interface
            messages = [
                {"role": "system", "content": self.format_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use adapter's query method directly
            response = await self.adapter.query_model(prompt)
            content = response.get("message", {}).get("content", "")
            
            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON in response")
            
            # Convert to FormattedContent
            sections = []
            for section_data in data.get("sections", []):
                sections.append(FormattedSection(
                    heading=section_data.get("heading", ""),
                    text=section_data.get("text", ""),
                    length=section_data.get("length", len(section_data.get("text", "")))
                ))
            
            return FormattedContent(
                url=scraped.url,
                title=data.get("title", scraped.title),
                metadata=data.get("metadata", {}),
                sections=sections,
                raw_html_snippet=scraped.html_content[:500] + "...",
                summary=data.get("summary", ""),
                content_length=scraped.content_length
            )
            
        except Exception as e:
            print(f"Model formatting failed: {e}")
            # Fall back to heuristic formatting
            fallback = self.create_fallback_format(scraped)
            fallback.format_error = str(e)
            return fallback
    
    async def format_content(self, scraped: ScrapedContent, use_model: bool = True) -> FormattedContent:
        """Main formatting function."""
        if not scraped.text_content and not scraped.html_content:
            return FormattedContent(
                url=scraped.url,
                title="Error",
                metadata={},
                sections=[],
                raw_html_snippet="",
                summary="Failed to scrape content",
                content_length=0,
                format_error=scraped.error or "No content"
            )
        
        if use_model:
            return await self.format_with_model(scraped)
        else:
            return self.create_fallback_format(scraped)
    
    async def format_multiple(self, scraped_list: List[ScrapedContent], 
                            use_model: bool = True) -> List[FormattedContent]:
        """Format multiple scraped contents."""
        if not scraped_list:
            return []
        
        print(f"Formatting {len(scraped_list)} pages...")
        
        tasks = [self.format_content(scraped, use_model) for scraped in scraped_list]
        results = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if result.format_error:
                print(f"  {i+1}/{len(scraped_list)}: {result.url} - ERROR: {result.format_error}")
            else:
                print(f"  {i+1}/{len(scraped_list)}: {result.url} - OK ({len(result.sections)} sections)")
        
        return results
    
    async def close(self):
        """Close adapter."""
        if self.adapter:
            await self.adapter.close()


async def main():
    """Test formatting functionality."""
    from scraper.playwright_scraper import ScrapedContent
    
    # Create test scraped content
    test_scraped = ScrapedContent(
        url="https://example.com",
        title="Example Page",
        text_content="""
        Welcome to Example
        
        This is the main content section. It contains useful information about the topic.
        Here is more detailed information that explains the concepts.
        
        Section Two: Advanced Topics
        
        This section covers more advanced material. It includes technical details
        and implementation guidelines.
        
        Conclusion
        
        This page provided an overview of the subject matter.
        """,
        html_content="<html><body><h1>Example</h1><p>Content...</p></body></html>",
        meta_tags={"description": "An example page"},
        og_tags={"og:title": "Example Page"},
        links=[],
        images=[],
        pdf_links=[],
        content_length=200,
        load_time=1.0,
        status_code=200
    )
    
    formatter = ContentFormatter()
    
    try:
        # Test formatting
        formatted = await formatter.format_content(test_scraped, use_model=False)
        
        print("Formatted Content:")
        print(f"Title: {formatted.title}")
        print(f"Summary: {formatted.summary}")
        print(f"Sections: {len(formatted.sections)}")
        
        for i, section in enumerate(formatted.sections):
            print(f"  {i+1}. {section.heading} ({section.length} chars)")
    
    finally:
        await formatter.close()


if __name__ == "__main__":
    asyncio.run(main())