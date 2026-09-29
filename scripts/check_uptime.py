#!/usr/bin/env python3
"""Dependency-free external checker for one or more WRS base URLs."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def check(base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/health"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "wrs-uptime-monitor/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
            if payload.get("status") != "ok":
                raise RuntimeError(f"reported status {payload.get('status')!r}")
            checks = payload.get("checks", {})
            required_checks = {"database", "openai", "embedding_model"}
            missing = sorted(required_checks - checks.keys())
            if missing:
                raise RuntimeError(f"missing checks: {', '.join(missing)}")
            failed = [name for name, item in checks.items() if item.get("status") not in {"ok", "skipped"}]
            if failed:
                raise RuntimeError(f"failed checks: {', '.join(failed)}")
            print(f"ok {url} version={payload.get('version', 'unknown')}")
            return
        except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"{url} failed after 3 attempts: {last_error}")


def main() -> int:
    urls = [arg for arg in sys.argv[1:] if arg.strip()]
    if not urls:
        print("No URLs configured", file=sys.stderr)
        return 2
    failures = []
    for url in urls:
        try:
            check(url)
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
