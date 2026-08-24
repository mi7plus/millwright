#!/usr/bin/env python3
"""Enforce the Cobertura branch-rate floor emitted by cargo-llvm-cov."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum", type=float, required=True)
    args = parser.parse_args()

    root = ET.parse(args.report).getroot()
    rate = float(root.attrib.get("branch-rate", "0")) * 100
    print(f"branch coverage: {rate:.2f}% (minimum: {args.minimum:.2f}%)")
    if rate < args.minimum:
        parser.error(f"branch coverage {rate:.2f}% is below {args.minimum:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
