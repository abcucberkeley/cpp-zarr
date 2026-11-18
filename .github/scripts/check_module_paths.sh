#!/bin/bash
# The Python module names none of the folders its headers come from (python/setup.py passes -ffile-prefix-map for
# them, so __FILE__ in pybind11's and Python's headers is relative): the temp folder, the Python include folders and
# pybind11's include folder, as module_prefix_maps.py lists them, must not appear in it followed by a separator, as
# a header's name has them (so that a bare "/tmp", which libstdc++ may hold, is no failure), on Windows in their /
# and their \ form with either separator after. A check that finds no folder to look for fails.
# usage: check_module_paths.sh <setup.py> <Python module>
set -u
setup=$1; mod=$2
[ -f "$mod" ] || { echo "no module at $mod"; exit 1; }
folders=()
while IFS= read -r f; do
    folders+=("$f")
done < <("${PYTHON:-python}" "$(dirname "$0")/module_prefix_maps.py" "$setup" --folders | tr -d '\r')
[ "${#folders[@]}" -gt 0 ] || { echo "no folders to check"; exit 1; }
forms=()
for f in "${folders[@]}"; do
    if command -v cygpath > /dev/null 2>&1; then
        # Either separator may follow either form (1.7.3's module held ...\pybind11\include/pybind11/...)
        for g in "$(cygpath -m "$f")" "$(cygpath -w "$f")"; do
            forms+=("$g/" "$g\\")
        done
    else
        forms+=("$f/")
    fi
done
bad=0
for p in "${forms[@]}"; do
    n=$(( $(grep -a -o -i -F -- "$p" "$mod" | wc -l) ))
    echo "$mod: $n strings with $p"
    if [ "$n" -gt 0 ]; then
        strings -a "$mod" | grep -i -F -- "$p" | head -5
        bad=1
    fi
done
exit $bad
