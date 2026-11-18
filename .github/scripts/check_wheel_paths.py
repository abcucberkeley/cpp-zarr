#!/usr/bin/env python3
# The wheel check under its old name, which release jobs call: check_release_paths.py, whose folder arguments here
# mean every *.whl in them, as before (a file argument may be any archive it reads).
# usage: check_wheel_paths.py [--forbid REGEX ...] <wheel or folder> ...
# exit: 0 if all is clean, 1 on any finding, 2 on bad arguments or a missing or unreadable wheel
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_release_paths  # noqa: E402

if __name__ == "__main__":
    sys.exit(check_release_paths.main(suffixes=(".whl",), noun="wheels"))
