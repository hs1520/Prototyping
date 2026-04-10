"""Quick connectivity check for calling AI with GitHub CLI authentication.

Usage:
    python tests/test_github.py
    python tests/test_github.py --prompt "Write a quicksort in Python"
    python tests/test_github.py --model openai/gpt-4.1-mini --show-json

Prerequisites:
- GitHub CLI installed: gh --version
- Logged in via GitHub CLI: gh auth login
- Account has access to GitHub Models/Copilot APIs

If authentication is missing, the script will guide you to log in and then retry once.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

import requests

DEFAULT_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODELS_ENDPOINT = "https://models.github.ai/catalog/models"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"


class GitHubAuthError(RuntimeError):
    """Raised when GitHub CLI authentication token cannot be obtained."""


def get_github_token() -> str:
    """Get an access token from GitHub CLI (gh auth token)."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitHubAuthError("GitHub CLI not found. Please install `gh` first.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        msg = "GitHub CLI is not authenticated. Run `gh auth login`."
        if stderr:
            msg = f"{msg} Details: {stderr}"
        raise GitHubAuthError(msg) from exc

    token = result.stdout.strip()
    if not token:
        raise GitHubAuthError("Empty token from `gh auth token`. Run `gh auth login` again.")
    return token


def call_github_models_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 60,
) -> dict[str, Any]:
    """Call GitHub Models chat completions endpoint with GitHub CLI token."""
    token = get_github_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    # Raise for 4xx/5xx so failures are explicit in CLI output.
    response.raise_for_status()
    return response.json()


def list_github_models(
    endpoint: str = DEFAULT_MODELS_ENDPOINT,
    timeout: int = 60,
) -> list[str]:
    """List model IDs currently available for the authenticated GitHub token."""
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.get(endpoint, headers=headers, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("data")
        items = raw_items if isinstance(raw_items, list) else []
    else:
        items = []

    model_ids = sorted(
        {
            item.get("id", "").strip()
            for item in items
            if isinstance(item, dict) and item.get("id")
        }
    )
    return model_ids


def filter_models_by_provider(model_ids: list[str], provider: str | None) -> list[str]:
    """Filter models by provider prefix, e.g. openai -> openai/*."""
    if not provider:
        return model_ids

    normalized = provider.strip().lower().rstrip("/")
    if not normalized:
        return model_ids

    prefix = f"{normalized}/"
    return [model_id for model_id in model_ids if model_id.lower().startswith(prefix)]


def extract_text(result: dict[str, Any]) -> str:
    """Extract assistant text from OpenAI-compatible chat response."""
    choices = result.get("choices")
    if not choices or not isinstance(choices, list):
        return ""

    first = choices[0] if choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = part.get("text")
                if isinstance(text_value, str):
                    text_chunks.append(text_value)
        return "\n".join(text_chunks).strip()

    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test AI call via GitHub CLI authentication (no standalone API key)."
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python quicksort function with a short explanation.",
        help="Prompt to send to the model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use.")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Chat completions endpoint (override for enterprise/proxy).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print full JSON response instead of extracted assistant text.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models for current GitHub token and exit.",
    )
    parser.add_argument(
        "--provider",
        default="",
        help="Optional provider filter for --list-models (e.g., openai, meta, mistral-ai).",
    )
    return parser.parse_args()


def prompt_user_login() -> bool:
    """Guide user through GitHub CLI login, then wait for confirmation."""
    print(
        "[AUTH] GitHub CLI is not authenticated. Please complete login to continue.",
        file=sys.stderr,
    )

    try:
        print("[AUTH] Trying to open interactive login via: gh auth login", file=sys.stderr)
        login_result = subprocess.run(["gh", "auth", "login"], check=False)
        if login_result.returncode != 0:
            print(
                "[AUTH] `gh auth login` did not finish successfully. "
                "You can run it manually and then continue.",
                file=sys.stderr,
            )
    except FileNotFoundError:
        print("[AUTH ERROR] GitHub CLI not found. Please install `gh` first.", file=sys.stderr)
        return False

    try:
        input("[AUTH] After login is complete, press Enter to retry (Ctrl+C to cancel): ")
    except KeyboardInterrupt:
        print("\n[AUTH] Cancelled by user.", file=sys.stderr)
        return False

    return True


def main() -> int:
    args = parse_args()
    has_retried_after_login = False
    result: dict[str, Any] | None = None

    if args.list_models:
        while True:
            try:
                model_ids = list_github_models(timeout=args.timeout)
                filtered = filter_models_by_provider(model_ids, args.provider)
                if args.show_json:
                    print(json.dumps(filtered, ensure_ascii=False, indent=2))
                else:
                    for model_id in filtered:
                        print(model_id)
                return 0
            except GitHubAuthError as exc:
                if has_retried_after_login:
                    print(f"[AUTH ERROR] {exc}", file=sys.stderr)
                    return 2
                if not prompt_user_login():
                    return 2
                has_retried_after_login = True
                continue
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                body = exc.response.text if exc.response is not None else str(exc)

                if status in (401, 403) and not has_retried_after_login:
                    print(f"[AUTH ERROR] status={status} body={body}", file=sys.stderr)
                    if not prompt_user_login():
                        return 2
                    has_retried_after_login = True
                    continue

                print(f"[HTTP ERROR] status={status} body={body}", file=sys.stderr)
                return 3
            except requests.RequestException as exc:
                print(f"[REQUEST ERROR] {exc}", file=sys.stderr)
                return 4

    while True:
        try:
            result = call_github_models_chat(
                prompt=args.prompt,
                model=args.model,
                endpoint=args.endpoint,
                timeout=args.timeout,
            )
            break
        except GitHubAuthError as exc:
            if has_retried_after_login:
                print(f"[AUTH ERROR] {exc}", file=sys.stderr)
                return 2
            if not prompt_user_login():
                return 2
            has_retried_after_login = True
            continue
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else str(exc)

            # Some environments return auth failures as 401/403 from the endpoint.
            if status in (401, 403) and not has_retried_after_login:
                print(f"[AUTH ERROR] status={status} body={body}", file=sys.stderr)
                if not prompt_user_login():
                    return 2
                has_retried_after_login = True
                continue

            print(f"[HTTP ERROR] status={status} body={body}", file=sys.stderr)
            return 3
        except requests.RequestException as exc:
            print(f"[REQUEST ERROR] {exc}", file=sys.stderr)
            return 4

    if result is None:
        print("[ERROR] No response received.", file=sys.stderr)
        return 5

    if args.show_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    text = extract_text(result)
    if text:
        print(text)
        return 0

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
