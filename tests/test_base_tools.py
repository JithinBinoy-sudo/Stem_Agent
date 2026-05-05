import pytest
from unittest.mock import patch, MagicMock
from tools.base_tools import web_search, url_reader, get_openai_schema

def test_web_search_returns_results():
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(title="Result 1", url="https://example.com", content="Some content")
    ]
    with patch("tools.base_tools.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = mock_response
        result = web_search(query="test query")
    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.com"

def test_web_search_handles_empty_results():
    mock_response = MagicMock()
    mock_response.results = []
    with patch("tools.base_tools.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = mock_response
        result = web_search(query="obscure query")
    assert result == {"results": []}

def test_url_reader_extracts_text():
    mock_html = "<html><body><p>Hello world</p></body></html>"
    with patch("tools.base_tools.requests.get") as mock_get:
        mock_get.return_value.text = mock_html
        mock_get.return_value.raise_for_status = MagicMock()
        result = url_reader(url="https://example.com")
    assert "Hello world" in result["text"]

def test_url_reader_handles_request_error():
    with patch("tools.base_tools.requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        result = url_reader(url="https://bad-url.com")
    assert "error" in result
    assert "Connection error" in result["error"]

def test_get_openai_schema_web_search():
    schema = get_openai_schema("web_search")
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]

def test_get_openai_schema_url_reader():
    schema = get_openai_schema("url_reader")
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "url_reader"
    assert "url" in schema["function"]["parameters"]["properties"]

def test_get_openai_schema_unknown_raises():
    with pytest.raises(KeyError):
        get_openai_schema("nonexistent_tool")
