from __future__ import annotations

import sys

from antigravityPicker.cli import main as picker_main


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]
    if "--latest-only" not in args:
        args.insert(0, "--latest-only")
    if "--force" not in args:
        args.insert(0, "--force")
    return picker_main(args)


if __name__ == "__main__":
    sys.exit(main())
