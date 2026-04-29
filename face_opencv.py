"""Convenience entry point for face detection, with camera enabled by default."""

from __future__ import annotations

import sys

from demo_opencv import main


def run() -> None:
    args = sys.argv[1:]
    normalized: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--mode":
            i += 2
            continue
        normalized.append(args[i])
        i += 1

    if "--input" not in normalized:
        normalized.extend(["--input", "0"])
    if "--display" not in normalized:
        normalized.append("--display")

    sys.argv = [sys.argv[0], "--mode", "face", *normalized]
    main()


if __name__ == "__main__":
    run()
