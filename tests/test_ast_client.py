"""Tests for the SysML AST HTTP client."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sysml.ast_client import SysMLAstClient
from src.sysml.ast_client import SysMLAstSchemaError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        return FakeResponse(
            {
                "schema_version": "1.0",
                "status": "ok",
                "ast": {"name": "DemoSystem"},
                "diagnostics": [],
            }
        )

    def get(self, url, timeout=None):
        self.calls.append((url, None, timeout))
        return FakeResponse({"status": "ok"})


class TestSysMLAstClient:
    def test_parse_text_posts_expected_payload(self):
        session = FakeSession()
        client = SysMLAstClient(
            base_url="http://localhost:8080",
            parse_path="/api/parse",
            parse_file_path="/api/parse-file",
            timeout_seconds=12,
            session=session,
        )

        result = client.parse_text("package DemoSystem {}", source_uri="file:///tmp/demo.sysml")

        assert result["schema_version"] == "1.0"
        assert result["status"] == "ok"
        assert result["ast"]["name"] == "DemoSystem"
        assert session.calls[0][0] == "http://localhost:8080/api/parse"
        assert session.calls[0][1]["sysml_text"] == "package DemoSystem {}"
        assert session.calls[0][1]["source_uri"] == "file:///tmp/demo.sysml"
        assert session.calls[0][2] == 12

    def test_parse_file_reads_and_posts(self, tmp_path: Path):
        file_path = tmp_path / "demo.sysml"
        file_path.write_text("package DemoSystem {}", encoding="utf-8")
        session = FakeSession()
        client = SysMLAstClient(base_url="http://localhost:8080", session=session)

        result = client.parse_file(file_path)

        assert result["schema_version"] == "1.0"
        assert result["status"] == "ok"
        assert result["ast"]["name"] == "DemoSystem"
        assert session.calls[0][0].endswith("/parse")
        assert session.calls[0][1]["sysml_text"] == "package DemoSystem {}"
        assert session.calls[0][1]["source_uri"].startswith("file://")

    def test_parse_text_raises_on_missing_schema_version(self):
        class _BadSession(FakeSession):
            def post(self, url, json=None, timeout=None):
                self.calls.append((url, json, timeout))
                return FakeResponse({"status": "ok", "ast": {"name": "DemoSystem"}})

        client = SysMLAstClient(base_url="http://localhost:8080", session=_BadSession())
        with pytest.raises(SysMLAstSchemaError):
            client.parse_text("package DemoSystem {}")

    def test_parse_text_raises_on_non_ok_status(self):
        class _BadSession(FakeSession):
            def post(self, url, json=None, timeout=None):
                self.calls.append((url, json, timeout))
                return FakeResponse(
                    {
                        "schema_version": "1.0",
                        "status": "error",
                        "error": "parse failure",
                        "ast": {},
                    }
                )

        client = SysMLAstClient(base_url="http://localhost:8080", session=_BadSession())
        with pytest.raises(SysMLAstSchemaError):
            client.parse_text("package DemoSystem {}")



