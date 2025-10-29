#!/usr/bin/env python3
"""
Async web scraper using Playwright for full content capture.
Handles timeouts, error recovery, and content extraction.
"""

import asyncio
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError


@dataclass
class ScrapedContent:
    """Structured scraped content."""
    url: str
    title: str
    text_content: str
    html_content: str
    meta_tags: Dict[str, str]
    og_tags: Dict[str, str]
    links: List[str]
    images: List[str]
    pdf_links: List[str]
    content_length: int
    load_time: float
    status_code: int
    error: Optional[str] = None


class PlaywrightScraper:
    """Async web scraper with Playwright."""
    
    def __init__(self, 
                 headless: bool = True,
                 timeout: int = 30,
                 max_concurrent: int = 3,
                 user_agent: str = None):
        self.headless = headless
        self.timeout = timeout * 1000  # Convert to milliseconds
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        self.browser: Optional[Browser] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Domains to avoid for safety
        self.blocked_domains = {
            'localhost', '127.0.0.1', '0.0.0.0',
            'facebook.com', 'instagram.com', 'twitter.com',  # Social (rate limited)
            'linkedin.com', 'pinterest.com'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the browser."""
        if self.browser:
            return
        
        try:
            playwright = await async_playwright().start()
            
            self.browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            print("✓ Playwright browser started")
        except Exception as e:
            print(f"⚠ Playwright browser not available: {e}")
            print("✓ Will use HTTP fallback for scraping")
            self.browser = None
    
    async def stop(self):
        """Stop the browser."""
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Warning: Error closing browser: {e}")
            finally:
                self.browser = None
    
    def is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed to be scraped."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain not in self.blocked_domains
            
        except Exception:
            return False
    
    async def create_page(self) -> Page:
        """Create a new page with proper configuration."""
        if not self.browser:
            await self.start()
        
        if not self.browser:
            raise Exception("Browser not available, using HTTP fallback")
        
        page = await self.browser.new_page()
        
        # Set user agent
        await page.set_user_agent(self.user_agent)
        
        # Set viewport
        await page.set_viewport_size({"width": 1920, "height": 1080})
        
        # Block unnecessary resources for speed
        await page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", 
                        lambda route: route.abort())
        
        return page
    
    def extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta tags from HTML."""
        meta_tags = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            
            if name and content:
                meta_tags[name.lower()] = content
        
        return meta_tags
    
    def extract_og_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract OpenGraph tags."""
        og_tags = {}
        
        for meta in soup.find_all('meta'):
            property_name = meta.get('property', '')
            if property_name.startswith('og:'):
                content = meta.get('content')
                if content:
                    og_tags[property_name] = content
        
        return og_tags
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith(('http://', 'https://')):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(base_url, href))
        
        return list(set(links))  # Remove duplicates
    
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs."""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            
            if src.startswith(('http://', 'https://')):
                images.append(src)
            elif src.startswith('/'):
                images.append(urljoin(base_url, src))
        
        return list(set(images))
    
    def extract_pdf_links(self, links: List[str]) -> List[str]:
        """Extract PDF links from all links."""
        return [link for link in links if link.lower().endswith('.pdf')]
    
    def clean_text_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean text content."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape a single URL."""
        start_time = time.time()
        
        # Check if URL is allowed
        if not self.is_url_allowed(url):
            return ScrapedContent(
                url=url,
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
                status_code=0,
                error="URL blocked for safety"
            )
        
        async with self.semaphore:
            page = None
            try:
                page = await self.create_page()
                
                # Navigate to URL
                response = await page.goto(url, timeout=self.timeout, wait_until="networkidle")
                
                if not response:
                    raise Exception("No response received")
                
                status_code = response.status
                
                # Wait for content to load
                await page.wait_for_load_state("networkidle", timeout=self.timeout)
                
                # Get page content
                title = await page.title()
                html_content = await page.content()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract various content types
                text_content = self.clean_text_content(soup)
                meta_tags = self.extract_meta_tags(soup)
                og_tags = self.extract_og_tags(soup)
                links = self.extract_links(soup, url)
                images = self.extract_images(soup, url)
                pdf_links = self.extract_pdf_links(links)
                
                load_time = time.time() - start_time
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    text_content=text_content,
                    html_content=html_content,
                    meta_tags=meta_tags,
                    og_tags=og_tags,
                    links=links,
                    images=images,
                    pdf_links=pdf_links,
                    content_length=len(text_content),
                    load_time=load_time,
                    status_code=status_code
                )
                
            except PlaywrightTimeoutError:
                return ScrapedContent(
                    url=url,
                    title="",
                    text_content="",
                    html_content="",
                    meta_tags={},
                    og_tags={},
                    links=[],
                    images=[],
                    pdf_links=[],
                    content_length=0,
                    load_time=time.time() - start_time,
                    status_code=0,
                    error="Timeout"
                )
                
            except Exception as e:
                return ScrapedContent(
                    url=url,
                    title="",
                    text_content="",
                    html_content="",
                    meta_tags={},
                    og_tags={},
                    links=[],
                    images=[],
                    pdf_links=[],
                    content_length=0,
                    load_time=time.time() - start_time,
                    status_code=0,
                    error=str(e)
                )
                
            finally:
                if page:
                    await page.close()
    
    async def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently."""
        if not urls:
            return []
        
        print(f"Scraping {len(urls)} URLs...")
        
        # Create tasks for concurrent scraping
        tasks = [self.scrape_url(url) for url in urls]
        
        # Run with progress
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if result.error:
                print(f"  {i+1}/{len(urls)}: {result.url} - ERROR: {result.error}")
            else:
                print(f"  {i+1}/{len(urls)}: {result.url} - OK ({result.content_length} chars)")
        
        return results


class HTTPFallbackScraper:
    """Fallback scraper using httpx when Playwright is not available."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape URL with HTTP client."""
        start_time = time.time()
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else ""
            text_content = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text_content.splitlines())
            text_content = ' '.join(line for line in lines if line)
            
            return ScrapedContent(
                url=url,
                title=title,
                text_content=text_content,
                html_content=response.text,
                meta_tags={},
                og_tags={},
                links=[],
                images=[],
                pdf_links=[],
                content_length=len(text_content),
                load_time=time.time() - start_time,
                status_code=response.status_code
            )
            
        except Exception as e:
            return ScrapedContent(
                url=url,
                title="",
                text_content="",
                html_content="",
                meta_tags={},
                og_tags={},
                links=[],
                images=[],
                pdf_links=[],
                content_length=0,
                load_time=time.time() - start_time,
                status_code=0,
                error=str(e)
            )
    
    async def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Scrape multiple URLs."""
        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


async def create_scraper(use_playwright: bool = True, **kwargs) -> PlaywrightScraper:
    """Factory function to create appropriate scraper."""
    try:
        if use_playwright:
            scraper = PlaywrightScraper(**kwargs)
            await scraper.start()
            return scraper
        else:
            return HTTPFallbackScraper()
    except Exception as e:
        print(f"Failed to create Playwright scraper: {e}")
        print("Falling back to HTTP scraper...")
        return HTTPFallbackScraper()


async def main():
    """Test scraping functionality."""
    test_urls = [
        "https://httpbin.org/html",  # Simple test page
        "https://example.com",       # Basic page
    ]
    
    async with PlaywrightScraper() as scraper:
        results = await scraper.scrape_urls(test_urls)
        
        for result in results:
            print(f"\nURL: {result.url}")
            print(f"Title: {result.title}")
            print(f"Status: {result.status_code}")
            print(f"Content length: {result.content_length}")
            print(f"Load time: {result.load_time:.2f}s")
            if result.error:
                print(f"Error: {result.error}")
            else:
                print(f"Text preview: {result.text_content[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())