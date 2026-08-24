#!/usr/bin/env python3
"""Keep package manifests and public documentation on one release version."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def cargo_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cargo.toml has no package version")
    return match.group(1)


def replace(path: str, pattern: str, replacement: str, *, check: bool) -> bool:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"expected version marker not found in {path}")
    if check:
        return updated == text
    if updated != text:
        target.write_text(updated, encoding="utf-8", newline="\n")
    return True


def sync(version: str, *, check: bool) -> list[str]:
    escaped = re.escape(version)
    checks = [
        ("Cargo.toml", r'(?m)^version = "[^"]+"', f'version = "{version}"'),
        (
            "Cargo.lock",
            r'(\[\[package\]\]\nname = "millwright"\nversion = ")[^"]+("\n)',
            rf'\g<1>{version}\2',
        ),
        ("pyproject.toml", r'(?m)^version = "[^"]+"', f'version = "{version}"'),
        ("index.html", r'<span class="ver">v[^<]+</span>', f'<span class="ver">v{version}</span>'),
        ("guide.html", r'<title>Millwright v[^<]+ · Guide</title>', f'<title>Millwright v{version} · Guide</title>'),
        ("guide.html", r'<meta name="millwright-version" content="[^"]+">', f'<meta name="millwright-version" content="{version}">'),
        ("guide.html", r'Millwright v[^ ]+ guide has moved', f'Millwright v{version} guide has moved'),
        ("docs/index.html", r'millwright = \{ version = <span class="s">"[^"]+"</span>', f'millwright = {{ version = <span class="s">"{version}"</span>'),
        ("tests/python_smoke.py", r'assert mw\.version\(\) == "[^"]+"', f'assert mw.version() == "{version}"'),
        ("tests/python/test_frame_and_profile.py", r'assert mw\.version\(\) == "[^"]+"', f'assert mw.version() == "{version}"'),
    ]
    stale = [path for path, pattern, value in checks if not replace(path, pattern, value, check=check)]

    if check:
        cargo = cargo_version()
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if not re.search(rf'(?m)^version = "{escaped}"$', pyproject):
            stale.append("pyproject.toml")
    return sorted(set(stale))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", help="release version (X.Y.Z)")
    parser.add_argument("--check", action="store_true", help="report drift without writing")
    args = parser.parse_args()
    version = args.version or cargo_version()
    if not SEMVER.fullmatch(version):
        parser.error("version must be X.Y.Z")
    stale = sync(version, check=args.check)
    if stale:
        print(f"version {version} is not synchronized in: {', '.join(stale)}", file=sys.stderr)
        return 1
    print(f"version {version} is synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
