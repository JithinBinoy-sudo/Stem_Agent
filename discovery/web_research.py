from tavily import TavilyClient
import config

SEARCH_QUERIES = [
    "{domain} AI agent architecture best practices",
    "{domain} agent tools and capabilities",
    "{domain} agent evaluation benchmarks",
]

class WebResearcher:
    def __init__(self):
        self._client = TavilyClient(api_key=config.TAVILY_API_KEY)

    def research(self, domain: str) -> list[dict]:
        findings = []
        for query_template in SEARCH_QUERIES:
            query = query_template.format(domain=domain)
            try:
                response = self._client.search(query=query, max_results=3)
                for result in response.results:
                    if result.content:
                        findings.append({
                            "source": result.url,
                            "title": result.title,
                            "insight": result.content[:500],
                            "query": query,
                        })
            except Exception:
                continue
        return findings
