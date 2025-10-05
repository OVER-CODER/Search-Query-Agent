from bs4 import BeautifulSoup
from typing import Dict, List
import re

def extract_best_snippet(html: str, query: str, max_chars: int = 800) -> Dict:
    if not html:
        return {"snippet": "", "score": 0}

    soup = BeautifulSoup(html, "lxml")
    meta_desc = soup.find("meta", {"property":"og:description"}) or soup.find("meta", {"name":"description"})
    if meta_desc and meta_desc.get("content"):
        snippet = meta_desc["content"]
        return {"snippet": snippet[:max_chars], "score": 0.7}

    text = ""
    candidates = []
    for tag_name in ["p", "li", "h1", "h2", "h3", "td"]:
        for el in soup.find_all(tag_name):
            t = el.get_text(separator=" ", strip=True)
            if t and len(t) > 30:
                candidates.append(t)

    q_tokens = set(re.findall(r"\w+", query.lower()))
    def score(text):
        tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(q_tokens & tokens)
        return overlap / (len(q_tokens) + 1)

    scored = sorted([(score(c), c) for c in candidates], key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        s = scored[0]
        return {"snippet": s[1][:max_chars], "score": float(s[0])}
    if candidates:
        return {"snippet": candidates[0][:max_chars], "score": 0.1}
    title = soup.title.string if soup.title else ""
    return {"snippet": (title or "")[:max_chars], "score": 0.05}