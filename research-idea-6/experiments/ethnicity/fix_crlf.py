#!/usr/bin/env python3
"""Fix Windows CRLF line endings on all experiment files."""
import os
import glob

base = "/workspace/experiments"
patterns = [
    os.path.join(base, "**", "*.py"),
    os.path.join(base, "**", "*.sh"),
    os.path.join(base, "**", "*.json"),
]
fixed = 0
for pat in patterns:
    for fpath in glob.glob(pat, recursive=True):
        with open(fpath, "rb") as f:
            data = f.read()
        if b"\r\n" in data:
            with open(fpath, "wb") as f:
                f.write(data.replace(b"\r\n", b"\n"))
            fixed += 1
            print(f"  Fixed: {fpath}")

print(f"Fixed {fixed} files")
