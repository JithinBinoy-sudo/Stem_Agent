import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from duckduckgo_search import DDGS
import config


def _search_tavily(query: str) -> dict:
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=5)
    return {
        "results": [
            {"title": r.title, "url": r.url, "content": r.content}
            for r in response.results
        ]
    }


def _search_duckduckgo(query: str) -> dict:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=5))
    return {
        "results": [
            {
                "title": h.get("title", ""),
                "url": h.get("href") or h.get("url", ""),
                "content": h.get("body", ""),
            }
            for h in hits
        ]
    }


def _search_wikipedia(query: str) -> dict:
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 5,
    }
    headers = {"User-Agent": "stem-agent/1.0 (educational)"}
    resp = requests.get(api, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for hit in data.get("query", {}).get("search", []):
        title = hit.get("title", "")
        snippet = BeautifulSoup(hit.get("snippet", ""), "html.parser").get_text()
        slug = title.replace(" ", "_")
        results.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{slug}",
            "content": snippet,
        })
    return {"results": results}


def web_search(query: str) -> dict:
    backend = getattr(config, "SEARCH_BACKEND", "duckduckgo").lower()
    try:
        if backend == "tavily":
            return _search_tavily(query)
        if backend == "wikipedia":
            return _search_wikipedia(query)
        primary = _search_duckduckgo(query)
        if primary.get("results"):
            return primary
        return _search_wikipedia(query)
    except Exception as e:
        try:
            fallback = _search_wikipedia(query)
            if fallback.get("results"):
                return fallback
        except Exception:
            pass
        return {"results": [], "error": str(e)}

def url_reader(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return {"url": url, "text": text[:4000]}
    except Exception as e:
        return {"url": url, "error": str(e)}

_SCHEMAS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    "url_reader": {
        "type": "function",
        "function": {
            "name": "url_reader",
            "description": "Read and extract text content from a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"}
                },
                "required": ["url"]
            }
        }
    }
}

def get_openai_schema(tool_name: str) -> dict:
    return _SCHEMAS[tool_name]
