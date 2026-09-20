#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Honest DFlash/GDN vs v1.1 speed capture. Never invents a 5x claim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studio" / "backend"))

from core.inference.helix_speed_policy import write_speed_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--candidate", type=float, default=None)
    parser.add_argument("--accepted-drafts", type=int, default=0)
    parser.add_argument("--error", default=None)
    args = parser.parse_args()
    payload = write_speed_json(
        Path(args.out),
        baseline_tok_s=args.baseline,
        candidate_tok_s=args.candidate,
        accepted_drafts=args.accepted_drafts,
        error=args.error,
    )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
