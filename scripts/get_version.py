#!/usr/bin/env python3

import os
import re

def get_version():
    """Get version from pyproject.toml."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "pyproject.toml")) as f:
        content = f.read()
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return version_match.group(1)

if __name__ == "__main__":
    print(get_version()) 