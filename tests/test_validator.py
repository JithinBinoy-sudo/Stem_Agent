import pytest
from safeguards.validator import Validator

validator = Validator()

SAFE_CODE = """
import re

def summarizer(text: str) -> dict:
    words = text.split()
    return {"summary": " ".join(words[:50]), "word_count": len(words)}
"""

DANGEROUS_OS_CODE = """
import os

def bad_tool(path: str) -> dict:
    os.system(f"rm -rf {path}")
    return {"done": True}
"""

DANGEROUS_EVAL_CODE = """
def bad_tool(expr: str) -> dict:
    result = eval(expr)
    return {"result": result}
"""

INVALID_SYNTAX_CODE = """
def broken(:
    pass
"""

CRASHING_CODE = """
def crash_tool(x: str) -> dict:
    raise RuntimeError("always fails")
"""

def test_safe_code_passes_all_gates():
    result = validator.validate(SAFE_CODE, mock_input={"text": "hello world"})
    assert result.passed is True
    assert result.error is None

def test_os_system_is_blocked():
    result = validator.validate(DANGEROUS_OS_CODE, mock_input={"path": "/tmp"})
    assert result.passed is False
    assert "dangerous" in result.error.lower()

def test_eval_is_blocked():
    result = validator.validate(DANGEROUS_EVAL_CODE, mock_input={"expr": "1+1"})
    assert result.passed is False
    assert "dangerous" in result.error.lower()

def test_invalid_syntax_fails():
    result = validator.validate(INVALID_SYNTAX_CODE, mock_input={})
    assert result.passed is False
    assert "syntax" in result.error.lower()

def test_crashing_code_fails_dry_run():
    result = validator.validate(CRASHING_CODE, mock_input={"x": "test"})
    assert result.passed is False
    assert "dry run" in result.error.lower()

def test_validation_result_has_stage():
    result = validator.validate(INVALID_SYNTAX_CODE, mock_input={})
    assert result.stage in ("syntax", "static_analysis", "dry_run")
