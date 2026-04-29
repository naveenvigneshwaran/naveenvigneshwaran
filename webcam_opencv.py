"""Convenience entry point for webcam object detection."""

from __future__ import annotations

import sys

from demo_opencv import main


def run() -> None:
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", "0", "--display"])
    main()


if __name__ == "__main__":
    run()
