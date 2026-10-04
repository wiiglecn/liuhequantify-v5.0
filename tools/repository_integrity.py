#!/usr/bin/env python3
"""Repository integrity checks: stable filenames and stale filename references."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSIONED_FILENAME = re.compile(
    r"(^|/)(?:[vV]\d+(?:[._-]?\d+)*|.*[_-]v\d+(?:[._-]?\d+)*)(?=\.[^.]+$)"
)
FORBIDDEN = (
    "v584_residual_feature_run.py",
    "v5841_statistical_hardening_run.py",
    "v5842_window_audit.py",
    "v57_meta_policy.py",
    "v55_1_signal_discovery_run.py",
    "v56_regime_adaptive_run.py",
    "v58_set_prediction_run.py",
    "test_v5841_statistical_hardening.py",
    "test_v5842_rolling_window.py",
)

def tracked_files():
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [Path(x) for x in out.decode().split("\0") if x]

def main():
    files = tracked_files()
    bad_names = [str(p) for p in files if VERSIONED_FILENAME.search(str(p))]
    text_files = [p for p in files if p.suffix.lower() in {".py", ".yml", ".yaml", ".md", ".txt", ".json", ".ini"}]
    stale = []
    for p in text_files:
        try:
            s = (ROOT / p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old in FORBIDDEN:
            if old in s:
                stale.append(f"{p}: {old}")
    if bad_names or stale:
        print("Repository integrity check FAILED")
        if bad_names:
            print("Versioned filenames:")
            print("\n".join(bad_names))
        if stale:
            print("Stale filename references:")
            print("\n".join(stale))
        return 1
    print(f"Repository integrity check PASSED: {len(files)} tracked files; 0 versioned filenames; 0 forbidden stale references.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
