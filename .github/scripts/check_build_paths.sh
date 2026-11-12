#!/bin/bash
# The installed library names no folder of the machine that built it (CMakeLists.txt passes
# -ffile-prefix-map for the source and build folders, so __FILE__ in the library's and its
# dependencies' sources is relative): the checkout's path must not appear in it, on Windows in its
# / and its \ form. On Windows a source named with \, as the release build's generator ("MinGW
# Makefiles") names the sources it compiles, is also checked to be remapped by a map given with /,
# as CMake gives it (GCC on Windows takes / and \ as the same in a map's folder).
# usage: check_build_paths.sh <checkout> <installed library> <build folder>
set -u
src=$1; lib=$2; build=$3
forms=("$src")
windows=0
if command -v cygpath > /dev/null 2>&1; then
    windows=1
    forms=("$(cygpath -m "$src")" "$(cygpath -w "$src")")
fi
bad=0
[ -f "$lib" ] || { echo "no library at $lib"; exit 1; }
for p in "${forms[@]}"; do
    n=$(( $(grep -a -o -i -F -- "$p" "$lib" | wc -l) ))
    echo "$lib: $n strings with $p"
    if [ "$n" -gt 0 ]; then
        strings -a "$lib" | grep -i -F -- "$p" | head -5
        bad=1
    fi
done
if [ "$windows" = 1 ]; then
    f="$build/prefix_map_check.c"
    printf 'const char *cppzarr_prefix_map_check = "PMC:" __FILE__;\n' > "$f"
    "${CC:-gcc}" -c "$(cygpath -w "$f")" "-ffile-prefix-map=$(cygpath -m "$build")=." -o "$build/prefix_map_check.o" || exit 1
    name=$(strings -a "$build/prefix_map_check.o" | grep -o 'PMC:.*' | head -1)
    name=${name#PMC:}
    echo "a source given as $(cygpath -w "$f") is named $name"
    if [ "$name" != "./prefix_map_check.c" ] && [ "$name" != '.\prefix_map_check.c' ]; then
        echo "not remapped"
        bad=1
    fi
fi
exit $bad
