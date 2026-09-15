#!/usr/bin/env python3
"""Run the dependency-light terminal Dockerfile validator.

Usage: dockerfile_precheck.py <task_dir_or_Dockerfile> [...]
Exit code 0 = all clean, 1 = at least one violation.
"""
from __future__ import annotations

import sys
from pathlib import Path

from harborrl.environments.terminal.validation import dockerfile_precheck_error


def main(argv: list[str]) -> int:
    bad = 0
    for arg in argv[1:]:
        p = Path(arg)
        dockerfile = p if p.name == "Dockerfile" else p / "Dockerfile"
        err = dockerfile_precheck_error(dockerfile)
        if err:
            bad += 1
            print(f"VIOLATION {err}")
        else:
            print(f"OK {dockerfile}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
