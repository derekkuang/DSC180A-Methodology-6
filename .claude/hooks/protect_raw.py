#!/usr/bin/env python3
"""PreToolUse hook on Bash: block shell writes into data/raw/.

The deny rules in settings.json cover the Edit and Write tools only, and the
PostToolUse hook only runs after Edit/Write. A `cp` from Bash replaced the raw
file with every test green (docs/SEEDED_FAULTS.md, fault 4). This runs before
the command, and exit 2 blocks it and tells the agent why.

It is a screen, not a proof: a Python script that opens the file for writing
gets past it. The sha256 pin in src/config.py (checked by `make data` and
tests/test_contract.py) catches that case after the fact.
"""
import json
import re
import shlex
import sys

RAW = "data/raw"
COPY_LIKE = {"cp", "mv", "install", "rsync", "ln"}          # destination is the last argument
WRITE_LIKE = {"rm", "tee", "truncate", "touch", "unlink"}    # any argument is written
RESTORE = re.compile(r"^git\s+(checkout|restore)\b")         # putting the committed bytes back is fine


def blocked(command: str) -> str | None:
    if RAW not in command:
        return None
    if re.search(r">>?\s*['\"]?[^\s'\"]*" + re.escape(RAW), command):
        return "redirects output into data/raw/"
    for part in re.split(r"&&|\|\||;|\|", command):
        part = part.strip()
        if not part or RESTORE.match(part):
            continue
        try:
            argv = shlex.split(part)
        except ValueError:
            argv = part.split()
        if not argv:
            continue
        prog, args = argv[0].rsplit("/", 1)[-1], argv[1:]
        paths = [a for a in args if not a.startswith("-")]
        if prog in COPY_LIKE and paths and RAW in paths[-1]:
            return f"`{prog}` writes into data/raw/"
        if prog in WRITE_LIKE and any(RAW in a for a in paths):
            return f"`{prog}` modifies a file in data/raw/"
        if prog == "sed" and any(a.startswith("-i") for a in args) and any(RAW in a for a in paths):
            return "`sed -i` edits a file in data/raw/ in place"
        if prog == "dd" and any(a.startswith("of=") and RAW in a for a in args):
            return "`dd of=` writes into data/raw/"
    return None


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    reason = blocked(payload.get("tool_input", {}).get("command", ""))
    if reason:
        print(
            f"Blocked: this command {reason}. Raw data is immutable; do any "
            "cleaning in src/ (downstream of load_raw). To undo a change, run "
            "`git checkout -- data/raw/`.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(0)
