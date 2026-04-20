"""Client for the external SysML v2 AST service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..config import Config


class SysMLAstClientError(RuntimeError):
    """Base error for AST client failures."""


class SysMLAstServiceError(SysMLAstClientError):
    """Raised when the AST service cannot be reached or returns an error."""


class SysMLAstSchemaError(SysMLAstClientError):
    """Raised when the AST service returns an invalid or unexpected schema."""


@dataclass
class SysMLAstClient:
    """Thin HTTP client for the Java AST service."""

    base_url: str = ""
    parse_path: str = ""
    parse_file_path: str = ""
    timeout_seconds: float = 30.0
    session: Any = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = Config.SYSML_AST_SERVICE_URL.strip()
        if not self.parse_path:
            self.parse_path = Config.SYSML_AST_PARSE_PATH
        if not self.parse_file_path:
            self.parse_file_path = Config.SYSML_AST_PARSE_FILE_PATH
        if not self.base_url:
            raise SysMLAstClientError(
                "SYSML_AST_SERVICE_URL is not configured; cannot call the AST service."
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise SysMLAstClientError(
                f"Invalid AST service URL: {self.base_url!r}"
            )
        if self.timeout_seconds <= 0:
            self.timeout_seconds = Config.SYSML_AST_TIMEOUT_SECONDS

    def parse_text(
        self,
        sysml_text: str,
        *,
        source_uri: str = "",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse SysML text into a JSON AST envelope."""
        if not sysml_text.strip():
            raise SysMLAstClientError("SysML text is empty.")

        payload = {
            "sysml_text": sysml_text,
            "source_uri": source_uri,
            "options": options or {},
        }
        return self._post_json(self._join_url(self.parse_path), payload)

    def parse_file(
        self,
        file_path: str | Path,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse a SysML file into a JSON AST envelope."""
        path = Path(file_path)
        if not path.exists():
            raise SysMLAstClientError(f"SysML file does not exist: {path}")
        sysml_text = path.read_text(encoding="utf-8")
        return self.parse_text(
            sysml_text,
            source_uri=path.as_uri(),
            options=options,
        )

    def health_check(self) -> bool:
        """Check whether the AST service is reachable."""
        try:
            response = self.session.get(
                self._join_url("/health"),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover - simple reachability helper
            raise SysMLAstServiceError(f"AST service health check failed: {exc}") from exc

    def _join_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            raise SysMLAstServiceError(f"AST service request failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise SysMLAstSchemaError(
                "AST service did not return valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise SysMLAstSchemaError(
                f"AST service returned {type(data).__name__}, expected JSON object."
            )
        return self._validate_envelope(data)

    def _validate_envelope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the finalized Java AST service schema envelope."""
        schema_version = data.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise SysMLAstSchemaError("AST payload missing non-empty 'schema_version'.")

        status = data.get("status")
        if status != "ok":
            details = data.get("error") or data.get("diagnostics") or "unknown error"
            raise SysMLAstSchemaError(f"AST payload status must be 'ok', got {status!r}: {details}")

        ast = data.get("ast")
        if not isinstance(ast, dict):
            raise SysMLAstSchemaError("AST payload missing object field 'ast'.")

        diagnostics = data.get("diagnostics")
        if diagnostics is not None and not isinstance(diagnostics, list):
            raise SysMLAstSchemaError("AST payload field 'diagnostics' must be a list when present.")

        return data




