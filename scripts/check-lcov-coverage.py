#!/usr/bin/env python3
"""Enforce line coverage for selected files in an LCOV report."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum", type=float, required=True)
    parser.add_argument("--file", action="append", required=True, dest="files")
    args = parser.parse_args()

    wanted = {Path(name).as_posix() for name in args.files}
    current: str | None = None
    found: set[str] = set()
    total = covered = 0
    for line in args.report.read_text(encoding="utf-8").splitlines():
        if line.startswith("SF:"):
            current = Path(line[3:]).as_posix()
            current = next((name for name in wanted if current.endswith(name)), None)
            if current:
                found.add(current)
        elif current and line.startswith("DA:"):
            count = int(line.split(",", 1)[1])
            total += 1
            covered += count > 0
        elif line == "end_of_record":
            current = None

    missing = wanted - found
    if missing:
        parser.error(f"LCOV report is missing: {', '.join(sorted(missing))}")
    rate = covered * 100 / total if total else 0.0
    print(f"Python binding coverage: {covered}/{total} lines ({rate:.2f}%)")
    if rate < args.minimum:
        parser.error(f"binding coverage {rate:.2f}% is below {args.minimum:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
