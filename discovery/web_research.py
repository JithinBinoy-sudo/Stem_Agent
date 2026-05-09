from tools.base_tools import web_search

SEARCH_QUERIES = [
    "{domain} AI agent architecture best practices",
    "{domain} agent tools and capabilities",
    "{domain} agent evaluation benchmarks",
]

class WebResearcher:
    def research(self, domain: str) -> list[dict]:
        findings = []
        for query_template in SEARCH_QUERIES:
            query = query_template.format(domain=domain)
            response = web_search(query=query)
            for result in response.get("results", []):
                content = result.get("content") or ""
                if content:
                    findings.append({
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                        "insight": content[:500],
                        "query": query,
                    })
        return findings
