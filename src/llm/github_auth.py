"""GitHub CLI authentication helpers for GitHub Models/Copilot access."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


class GitHubCLIAuthError(RuntimeError):
    """Raised when GitHub CLI authentication cannot provide a usable token."""


class GitHubAuthManager:
    """Manage token retrieval and default GitHub CLI login flow."""

    def __init__(self, auto_login: bool = True):
        self.auto_login = auto_login

    def get_token(self, explicit_token: Optional[str] = None) -> Optional[str]:
        """Return a usable token from explicit/env/CLI sources, with optional auto-login."""
        token = explicit_token or self._get_env_token() or self._get_github_cli_token(raise_on_error=False)
        if token:
            return token

        if not self.auto_login:
            return None

        return self._run_default_github_login()

    def refresh_token(self) -> str:
        """Force default login flow and return a fresh token."""
        if not self.auto_login:
            raise GitHubCLIAuthError(
                "Auto login is disabled and token refresh is unavailable."
            )
        return self._run_default_github_login()

    @staticmethod
    def _get_env_token() -> Optional[str]:
        for key in ("GITHUB_TOKEN", "GH_TOKEN"):
            value = (os.environ.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _get_github_cli_token(raise_on_error: bool = True) -> Optional[str]:
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            if raise_on_error:
                raise GitHubCLIAuthError(
                    "GitHub CLI not found. Please install `gh` first."
                ) from exc
            return None
        except subprocess.CalledProcessError as exc:
            if raise_on_error:
                stderr = (exc.stderr or "").strip()
                detail = f" Details: {stderr}" if stderr else ""
                raise GitHubCLIAuthError(
                    "GitHub CLI is not authenticated. Run `gh auth login`." + detail
                ) from exc
            return None

        token = result.stdout.strip()
        if token:
            return token

        if raise_on_error:
            raise GitHubCLIAuthError("Empty token returned from `gh auth token`.")
        return None

    def _run_default_github_login(self) -> str:
        print(
            "[AUTH] GitHub login required. Starting browser login with defaults: "
            "github.com + https + setup-git.",
            file=sys.stderr,
        )
        try:
            login_result = subprocess.run(
                [
                    "gh",
                    "auth",
                    "login",
                    "--hostname",
                    "github.com",
                    "--git-protocol",
                    "https",
                    "--web",
                ],
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitHubCLIAuthError("GitHub CLI not found. Please install `gh` first.") from exc

        if login_result.returncode != 0:
            raise GitHubCLIAuthError(
                "`gh auth login` failed. Please run it manually and retry."
            )

        subprocess.run(["gh", "auth", "setup-git"], check=False)

        token = self._get_github_cli_token(raise_on_error=False)
        if not token:
            raise GitHubCLIAuthError("Login finished but no token is available from `gh auth token`.")
        return token

    @staticmethod
    def is_auth_error(exc: Exception) -> bool:
        """Detect authentication failures from OpenAI-compatible client errors."""
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return True

        text = str(exc).lower()
        return "authentication" in text or "unauthorized" in text or "invalid api key" in text
