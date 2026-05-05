from tools.base_tools import web_search, url_reader

TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
    "url_reader": url_reader,
}

def get_tool(name: str):
    return TOOL_REGISTRY[name]

def register_tool(name: str, func: callable):
    TOOL_REGISTRY[name] = func

def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())
