#!/bin/bash
# The installed library exports none of its static C++ runtime and none of the nlohmann/json code it builds in
# (CMakeLists.txt's CPPZARR_PRIVATE_RUNTIME), counted by demangled name, and still exports its API: the zarr class
# and the functions of its installed headers (checked by demangled name too, so that names c++filt could not
# demangle fail the check rather than pass it). A Python module given second exports no C++ symbol at all (macOS,
# where python/setup.py links it with its entry point alone exported). Linux and macOS. (cpp-tiff has the same check.)
# On macOS the names are given to c++filt as nm prints them, with the one more leading _ of Mach-O names, which
# Apple's c++filt drops itself: given without it (_ZN4zarrC1Ev), it left every C++ name mangled.
# usage: check_exports.sh <installed library> [<Python module>]
set -u
lib=$1; mod=${2:-}
case "$(uname -s)" in
    Darwin)
        exports(){ nm -gU "$1" | awk '{print $NF}'; }
        cxx='^__Z'
        if [ "$(echo __ZN4zarrC1Ev | c++filt)" = 'zarr::zarr()' ]; then demangle(){ c++filt; }
        else demangle(){ c++filt -_; }; fi ;;
    *)
        exports(){ nm -D --defined-only "$1" | awk '{print $NF}' | sed 's/@.*//'; }
        cxx='^_Z'
        demangle(){ c++filt; } ;;
esac
[ -f "$lib" ] || { echo "no library at $lib"; exit 1; }
bad=0
all=$(exports "$lib")
names=$(printf '%s\n' "$all" | demangle)
ns='(std::|__gnu_cxx::|__cxxabiv1::|nlohmann::)'
builtin='(unsigned |signed )?(char|short|int|long|long long|bool|float|double|long double|void|wchar_t|char8_t|char16_t|char32_t|decltype\(nullptr\)|__int128|__float128)( const)?\*?'
# (the C ABI's names keep their Mach-O _ on macOS)
pattern="^($ns|operator new|operator delete|(typeinfo|typeinfo name|vtable|VTT|construction vtable|guard variable|TLS init function|TLS wrapper function) for ($ns|$builtin\$)|(non-)?virtual thunk to $ns|covariant return thunk to $ns|_?__cxa_|_?__gxx_personality|_?__dynamic_cast\$)"
# The check checks itself: names of the runtime as nm prints them here (a destructor of std's, a virtual thunk to
# a member of one of its short forms, the typeinfo of a built-in pointer type and a function of the C ABI) are found
# as the runtime's, and a destructor of std's demangles (cpp-tiff's check does the same)
probes='_ZNSt9exceptionD1Ev _ZTv0_n24_NSoD1Ev _ZTIPKc __cxa_throw'
[ "$(uname -s)" = Darwin ] && probes=$(for p in $probes; do printf '_%s ' "$p"; done)
probed=$(printf '%s\n' $probes | demangle)
found=$(printf '%s\n' "$probed" | grep -c -E "$pattern" || true)
if [ "$found" != 4 ] || [ "$(printf '%s\n' "$probed" | head -1)" != 'std::exception::~exception()' ]; then
    echo "the check finds $found of 4 of the runtime's names, demangled as: $(printf '%s\n' "$probed" | tr '\n' ';')"
    bad=1
fi
runtime=$(printf '%s\n' "$names" | grep -E "$pattern")
n=$(printf '%s' "$runtime" | grep -c . || true)
echo "$lib: $(printf '%s\n' "$all" | grep -c .) exports, $n of the C++ runtime or nlohmann/json"
if [ "$n" -gt 0 ]; then printf '%s\n' "$runtime" | head -10; bad=1; fi
for api in 'zarr::zarr(' 'zarr::get_shape(' 'parallelReadZarr(' 'parallelWriteZarr(' 'releaseBuffers()'; do
    printf '%s\n' "$names" | cut -c1-${#api} | grep -qxF -- "$api" || { echo "$lib does not export $api"; bad=1; }
done
if [ -n "$mod" ]; then
    [ -f "$mod" ] || { echo "no module at $mod"; exit 1; }
    n=$(exports "$mod" | grep -c "$cxx" || true)
    echo "$mod: $(exports "$mod" | grep -c .) exports, $n of C++"
    if [ "$n" -gt 0 ]; then exports "$mod" | grep "$cxx" | demangle | head -10; bad=1; fi
fi
exit $bad
