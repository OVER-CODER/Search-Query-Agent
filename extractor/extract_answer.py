#!/usr/bin/env python3
"""
Answer extraction from formatted content using local model.
Finds exact answers to user queries from structured page data.
"""

import json
import asyncio
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from server.ollama_adapter import OllamaAdapter, create_adapter
from formatter.format_with_model import FormattedContent


@dataclass
class ExtractedAnswer:
    """Extracted answer with metadata."""
    answer: str
    confidence: float
    source_url: str
    source_title: str
    source_section: str
    reasoning: str
    relevant_text: str


class AnswerExtractor:
    """Extract specific answers from formatted content."""
    
    def __init__(self, adapter: OllamaAdapter = None):
        self.adapter = adapter or create_adapter()
        
        # System prompt for answer extraction
        self.extract_prompt = """You are an answer extraction assistant.
Given a user query and structured webpage content, extract the exact answer.

Response format:
{
  "answer": "The specific answer to the query",
  "confidence": 0.95,
  "reasoning": "Why this answer is correct",
  "relevant_text": "The specific text that contains the answer"
}

Rules:
- Only extract answers that directly address the query
- Be specific and factual
- If no clear answer exists, set confidence to 0.1 and answer to "No specific answer found"
- Quote the exact relevant text that supports your answer
- Confidence should reflect how certain the answer is (0.0 to 1.0)

Respond with valid JSON only."""
    
    def create_content_context(self, formatted: FormattedContent, max_chars: int = 3000) -> str:
        """Create context string from formatted content."""
        context_parts = [
            f"Title: {formatted.title}",
            f"URL: {formatted.url}",
            f"Summary: {formatted.summary}",
            "",
            "Content Sections:"
        ]
        
        char_count = len('\n'.join(context_parts))
        
        for section in formatted.sections:
            section_text = f"\n{section.heading}:\n{section.text}"
            
            if char_count + len(section_text) > max_chars:
                break
            
            context_parts.append(section_text)
            char_count += len(section_text)
        
        return '\n'.join(context_parts)
    
    def extract_answer_heuristic(self, query: str, formatted: FormattedContent) -> ExtractedAnswer:
        """Extract answer using simple heuristics (fallback)."""
        query_words = set(query.lower().split())
        best_section = None
        best_score = 0
        
        # Find section with most query word matches
        for section in formatted.sections:
            section_words = set(section.text.lower().split())
            overlap = len(query_words.intersection(section_words))
            score = overlap / len(query_words) if query_words else 0
            
            if score > best_score:
                best_score = score
                best_section = section
        
        if best_section and best_score > 0.2:
            # Extract relevant sentences
            sentences = re.split(r'[.!?]+', best_section.text)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_words.intersection(sentence_words):
                    relevant_sentences.append(sentence.strip())
            
            answer_text = ' '.join(relevant_sentences[:2])  # Take first 2 relevant sentences
            
            return ExtractedAnswer(
                answer=answer_text[:300] + "..." if len(answer_text) > 300 else answer_text,
                confidence=min(best_score * 2, 0.8),  # Max 0.8 for heuristic
                source_url=formatted.url,
                source_title=formatted.title,
                source_section=best_section.heading,
                reasoning=f"Found {len(relevant_sentences)} relevant sentences in section '{best_section.heading}'",
                relevant_text=answer_text
            )
        else:
            return ExtractedAnswer(
                answer="No specific answer found in this content",
                confidence=0.1,
                source_url=formatted.url,
                source_title=formatted.title,
                source_section="",
                reasoning="No matching content found",
                relevant_text=""
            )
    
    async def extract_with_model(self, query: str, formatted: FormattedContent) -> ExtractedAnswer:
        """Extract answer using the local model."""
        try:
            # Create context from formatted content
            context = self.create_content_context(formatted)
            
            # Create extraction prompt
            prompt = f"""Extract the exact answer to this query from the provided content:

Query: {query}

Content:
{context}

Please analyze the content and extract the specific answer that addresses the query."""
            
            # Query model
            messages = [
                {"role": "system", "content": self.extract_prompt},
                {"role": "user", "content": prompt}
            ]
            
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
            
            return ExtractedAnswer(
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.0)),
                source_url=formatted.url,
                source_title=formatted.title,
                source_section="AI Analysis",
                reasoning=data.get("reasoning", ""),
                relevant_text=data.get("relevant_text", "")
            )
            
        except Exception as e:
            print(f"Model extraction failed: {e}")
            # Fall back to heuristic extraction
            fallback = self.extract_answer_heuristic(query, formatted)
            fallback.reasoning += f" (Model error: {str(e)})"
            return fallback
    
    async def extract_answer(self, query: str, formatted: FormattedContent, 
                           use_model: bool = True) -> ExtractedAnswer:
        """Main answer extraction function."""
        if not formatted.sections:
            return ExtractedAnswer(
                answer="No content available for analysis",
                confidence=0.0,
                source_url=formatted.url,
                source_title=formatted.title,
                source_section="",
                reasoning="No content sections found",
                relevant_text=""
            )
        
        if use_model:
            return await self.extract_with_model(query, formatted)
        else:
            return self.extract_answer_heuristic(query, formatted)
    
    async def extract_from_multiple(self, query: str, formatted_list: List[FormattedContent],
                                  use_model: bool = True) -> List[ExtractedAnswer]:
        """Extract answers from multiple formatted contents."""
        if not formatted_list:
            return []
        
        print(f"Extracting answers for query: '{query}' from {len(formatted_list)} pages...")
        
        tasks = [self.extract_answer(query, formatted, use_model) for formatted in formatted_list]
        results = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            print(f"  {i+1}/{len(formatted_list)}: {result.source_url} - "
                  f"Confidence: {result.confidence:.2f}")
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def combine_answers(self, answers: List[ExtractedAnswer], 
                       min_confidence: float = 0.5) -> Optional[ExtractedAnswer]:
        """Combine multiple answers into a best answer."""
        if not answers:
            return None
        
        # Filter by minimum confidence
        good_answers = [a for a in answers if a.confidence >= min_confidence]
        
        if not good_answers:
            # Return best answer even if low confidence
            return answers[0]
        
        # If we have good answers, take the best one
        best_answer = good_answers[0]
        
        # If multiple good answers, combine information
        if len(good_answers) > 1:
            additional_sources = [f"{a.source_title} ({a.confidence:.2f})" 
                                for a in good_answers[1:3]]  # Top 3 total
            
            best_answer.reasoning += f" Also supported by: {', '.join(additional_sources)}"
        
        return best_answer
    
    async def close(self):
        """Close adapter."""
        if self.adapter:
            await self.adapter.close()


async def main():
    """Test answer extraction."""
    from formatter.format_with_model import FormattedContent, FormattedSection
    
    # Create test formatted content
    test_formatted = FormattedContent(
        url="https://example.com/python",
        title="Python Programming Guide",
        metadata={},
        sections=[
            FormattedSection(
                heading="What is Python?",
                text="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes readability and simplicity.",
                length=120
            ),
            FormattedSection(
                heading="Key Features",
                text="Python features include dynamic typing, automatic memory management, and extensive standard library. It supports multiple programming paradigms.",
                length=140
            )
        ],
        raw_html_snippet="<html>...</html>",
        summary="Guide to Python programming language",
        content_length=260
    )
    
    extractor = AnswerExtractor()
    
    try:
        # Test extraction
        answer = await extractor.extract_answer(
            "Who created Python?", 
            test_formatted, 
            use_model=False
        )
        
        print("Extracted Answer:")
        print(f"Query: Who created Python?")
        print(f"Answer: {answer.answer}")
        print(f"Confidence: {answer.confidence:.2f}")
        print(f"Reasoning: {answer.reasoning}")
        print(f"Source: {answer.source_title}")
    
    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())