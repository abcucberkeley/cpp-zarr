#!/bin/bash
# Checks that the mex files given run no OpenMP of their own: their conversions are the library's
# (helperfunctions.h). On Windows each mex file links a private OpenMP runtime, whose threads, once a loop
# of the mex file's own had started them, kept it loaded after MATLAB released it, and the last of them to
# end unloaded it, holding the loader lock, which hung MATLAB as it exited. Looks for the outlined loops
# (._omp_fn.) and the runtime's entry points (GOMP_parallel...) among each mex file's symbols.
# The check tests itself first: the same search must find them in an object compiled from an OpenMP loop
# on this system, and each mex file's symbols must include mexFunction, so that a search that stopped
# matching, or a mex file without its symbols, fails the check instead of passing it.
# usage: check_mex_openmp.sh <mex file>...   (CXX: the compiler, g++ by default)
set -u
pattern='_omp_fn\.|GOMP_parallel|gomp_thread_start'
# (a file's symbol names: nm -D for a Linux shared object's exports, nm for the others)
syms(){ { nm "$1" 2>/dev/null; nm -D "$1" 2>/dev/null; } | awk 'NF {print $NF}'; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'int f(int n){\n    int s = 0;\n    #pragma omp parallel for reduction(+:s)\n    for(int i = 0; i < n; i++) s += i;\n    return s;\n}\n' > "$tmp/probe.cpp"
"${CXX:-g++}" -fopenmp -O2 -c "$tmp/probe.cpp" -o "$tmp/probe.o" || { echo "could not compile an OpenMP loop with ${CXX:-g++}"; exit 1; }
found=$(syms "$tmp/probe.o" | grep -c -E "$pattern" || true)
if [ "$found" -lt 2 ]; then
    echo "the search finds $found OpenMP symbols in an OpenMP loop's object (2 or more expected), whose symbols are:"
    syms "$tmp/probe.o"
    exit 1
fi
[ $# -gt 0 ] || { echo "no mex files given"; exit 1; }
bad=0
for f in "$@"; do
    s=$(syms "$f")
    if ! printf '%s\n' "$s" | grep -qE '^_?mexFunction(@.*)?$'; then
        echo "$f: no mexFunction among its symbols, so the check cannot tell"
        bad=1
        continue
    fi
    omp=$(printf '%s\n' "$s" | grep -E "$pattern" | sort -u)
    if [ -n "$omp" ]; then
        echo "$f runs OpenMP of its own:"
        printf '%s\n' "$omp" | head -10
        bad=1
    else
        echo "$f: no OpenMP of its own"
    fi
done
exit $bad
