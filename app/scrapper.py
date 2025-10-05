# app/scraper.py
import asyncio
from typing import Dict, List, Tuple
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 15000  # ms

async def fetch_with_playwright(url: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, Dict]:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (compatible; ResearchAgent/1.0)")
            page = await context.new_page()
            await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("domcontentloaded")
            content = await page.content()
            meta = {
                "status": 200,
                "title": await page.title()
            }
            await context.close()
            await browser.close()
            return content, meta
    except PlaywrightTimeoutError:
        return "", {"status": 408, "error": "timeout"}
    except Exception as e:
        return "", {"status": 520, "error": str(e)}

async def fetch_with_httpx(url: str, timeout: int = 10) -> Tuple[str, Dict]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            r = await client.get(url)
            return r.text, {"status": r.status_code, "title": ""}
    except Exception as e:
        return "", {"status": 520, "error": str(e)}

async def scrape_url(url: str) -> Dict:
    html, meta = await fetch_with_playwright(url)
    if not html:
        html, meta = await fetch_with_httpx(url)
    return {
        "url": url,
        "html": html,
        "title": meta.get("title", ""),
        "status": meta.get("status", 0),
        "error": meta.get("error")
    }

async def scrape_urls_concurrent(urls: List[str], concurrency: int = 3) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def sem_task(u):
        async with sem:
            return await scrape_url(u)

    tasks = [asyncio.create_task(sem_task(u)) for u in urls]
    for t in asyncio.as_completed(tasks):
        res = await t
        results.append(res)
    return results



async def main():
    urls_input = input("Enter URLs separated by commas:\n> ")
    urls = [u.strip() for u in urls_input.split(",") if u.strip()]

    print(f"\n🔍 Fetching {len(urls)} URLs concurrently...\n")
    results = await scrape_urls_concurrent(urls, concurrency=3)

    print("\n✅ Full Scraped Results:\n")
    for r in results:
        print(f"URL: {r['url']}")
        print(f"Status: {r['status']}")
        print(f"Title: {r['title']}")
        if r.get("error"):
            print(f"Error: {r['error']}")
        print("\n----- HTML Content Start -----\n")
        print(r['html'][:5000])
        print("\n----- HTML Content End -----\n")
        print("-" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
