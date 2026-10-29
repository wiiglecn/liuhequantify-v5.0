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
    bad_names = [str(path) for path in files if VERSIONED_FILENAME.search(str(path))]
    text_files = [
        path for path in files
        if path.suffix.lower() in {".py", ".yml", ".yaml", ".md", ".txt", ".json", ".ini"}
        # 审计规则自身必须包含被检测的旧文件名，因此不能把自身当作被审计内容。
        and path.as_posix() != "tools/repository_integrity.py"
    ]
    stale = []
    for path in text_files:
        try:
            content = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old_name in FORBIDDEN:
            if old_name in content:
                stale.append(f"{path}: {old_name}")

    if bad_names or stale:
        print("Repository integrity check FAILED")
        if bad_names:
            print("Versioned filenames:")
            print("\n".join(bad_names))
        if stale:
            print("Stale filename references:")
            print("\n".join(stale))
        return 1

    print(
        f"Repository integrity check PASSED: {len(files)} tracked files; "
        "0 versioned filenames; 0 forbidden stale references."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
