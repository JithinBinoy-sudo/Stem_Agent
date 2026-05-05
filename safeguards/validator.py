import ast
import importlib.util
import sys
import tempfile
import os
from dataclasses import dataclass
from typing import Optional
import config

@dataclass
class ValidationResult:
    passed: bool
    stage: str
    error: Optional[str] = None

class Validator:
    def validate(self, code: str, mock_input: dict) -> ValidationResult:
        result = self._syntax_check(code)
        if not result.passed:
            return result
        result = self._static_analysis(code)
        if not result.passed:
            return result
        return self._dry_run(code, mock_input)

    def _syntax_check(self, code: str) -> ValidationResult:
        try:
            ast.parse(code)
            return ValidationResult(passed=True, stage="syntax")
        except SyntaxError as e:
            return ValidationResult(passed=False, stage="syntax", error=f"syntax error: {e}")

    def _static_analysis(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(passed=False, stage="static_analysis", error="syntax error during analysis")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in config.DANGEROUS_IMPORTS:
                        return ValidationResult(
                            passed=False, stage="static_analysis",
                            error=f"dangerous import detected: {alias.name}"
                        )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in config.DANGEROUS_IMPORTS:
                    return ValidationResult(
                        passed=False, stage="static_analysis",
                        error=f"dangerous import detected: {module}"
                    )
            if isinstance(node, ast.Call):
                # Check direct calls like eval(...), exec(...), __import__(...)
                if isinstance(node.func, ast.Name) and node.func.id in config.DANGEROUS_IMPORTS:
                    return ValidationResult(
                        passed=False, stage="static_analysis",
                        error=f"dangerous call detected: {node.func.id}"
                    )
            # Check attribute access calls like os.system(...), shutil.rmtree(...)
            # This fixes a plan bug: DANGEROUS_OS_CODE uses os.system which is an
            # ast.Attribute node, not ast.Name, so must be caught separately.
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    full = f"{node.value.id}.{node.attr}"
                    if full in config.DANGEROUS_IMPORTS:
                        return ValidationResult(
                            passed=False, stage="static_analysis",
                            error=f"dangerous attribute access detected: {full}"
                        )

        return ValidationResult(passed=True, stage="static_analysis")

    def _dry_run(self, code: str, mock_input: dict) -> ValidationResult:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name

            spec = importlib.util.spec_from_file_location("_dry_run_module", tmp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func = None
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                return ValidationResult(passed=False, stage="dry_run", error="dry run: no callable function found")

            func(**mock_input)
            return ValidationResult(passed=True, stage="dry_run")
        except Exception as e:
            return ValidationResult(passed=False, stage="dry_run", error=f"dry run failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
