#!/usr/bin/env python3
"""Compatibility entry point for shared Colab helpers."""
from pathlib import Path
import sys
for _root in Path(__file__).resolve().parents:
    if (_root / "media_workspace").is_dir():
        sys.path.insert(0, str(_root))
        break
from media_workspace.colab_frontend import *

if __name__ == "__main__":
    raise SystemExit(main())
