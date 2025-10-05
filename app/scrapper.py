import asyncio
import json
from typing import Dict, List, Tuple, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 15000

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

def parse_general_content(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    content = {
        "headings": [],
        "paragraphs": [],
        "lists": [],
        "tables": [],
        "links": [],
        "ids": []
    }

    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            content["headings"].append({"level": level, "text": h.get_text(strip=True)})

    for p in soup.find_all("p"):
        content["paragraphs"].append(p.get_text(strip=True))

    for ul in soup.find_all(["ul", "ol"]):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        if items:
            content["lists"].append(items)

    for table in soup.find_all("table"):
        table_data = []
        rows = table.find_all("tr")
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])] if rows else []
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if headers and len(cells) == len(headers):
                table_data.append(dict(zip(headers, cells)))
            else:
                table_data.append(cells)
        if table_data:
            content["tables"].append(table_data)

    for a in soup.find_all("a", href=True):
        content["links"].append({"text": a.get_text(strip=True), "href": a["href"]})

    for tag in soup.find_all(attrs={"id": True}):
        content["ids"].append(tag["id"])

    return content

async def main():
    urls_input = input("Enter URLs separated by commas:\n> ")
    urls = [u.strip() for u in urls_input.split(",") if u.strip()]

    print(f"\n🔍 Fetching {len(urls)} URLs concurrently...\n")
    results = await scrape_urls_concurrent(urls, concurrency=3)

    full_data = []
    for r in results:
        entry = {
            "url": r["url"],
            "status": r["status"],
            "title": r["title"],
            "error": r.get("error"),
            "structured_content": parse_general_content(r["html"]) if r.get("html") else {}
        }
        full_data.append(entry)

    # Store to JSON file
    output_file = "scraped_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
