#!/usr/bin/env python3
"""Run the shared SPilot/Slime profile summarizer for TMax runs."""

from __future__ import annotations

from pathlib import Path
import runpy


COMMON_SUMMARIZER = (
    Path(__file__).resolve().parents[2]
    / "spilot_router_slime_grpo"
    / "profile"
    / "profile_summary.py"
)

if __name__ == "__main__":
    runpy.run_path(str(COMMON_SUMMARIZER), run_name="__main__")
