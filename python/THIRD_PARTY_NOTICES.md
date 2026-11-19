# Third-party notices

cpp-zarr is BSD 3-Clause (`LICENSE`). The works below are built into it or ship with it; their license files, copied unchanged from the sources named, are under `licenses/`.

Where they end up:

- **libcppZarr**: the library (`libcppZarr.so`, `libcppZarr.dylib`, `libcppZarr.dll`), in every wheel and every MATLAB folder.
- **the module**: the Python extension module `cppzarr.cppzarr`, in every wheel.
- **Linux wheels**: manylinux_2_28 x86_64, built with the image's gcc-toolset-14 (GCC 14.2.1); they bundle `cpp_zarr.libs/libcppZarr-*.so` and `cpp_zarr.libs/libgomp-*.so.1.0.0`.
- **macOS wheels**: arm64, built with Homebrew's GCC 13; they bundle `cppzarr/.dylibs/libcppZarr.dylib` and `cppzarr/.dylibs/libgomp.1.dylib`.
- **Windows wheels**: x86_64, built with MinGW-w64's GCC 12.2; they hold `cppzarr/libcppZarr.dll` and bundle `cpp_zarr.libs/libgomp-1.dll`, `libgcc_s_seh-1*.dll` and `libwinpthread-1.dll`.
- **MATLAB folders**, each also in allMatlab.tar.gz: `linux/` (libcppZarr.so, libgomp.so.1, the mex files), `macArm/` (libcppZarr.dylib, libgomp.1.dylib, libgcc_s.1.1.0.dylib, libstdc++.6.0.32.dylib, the mex files) and `windows/` (libcppZarr.dll, libgomp-1.dll, libgcc_s_seh-1.dll, libwinpthread-1.dll, the mex files).

## c-blosc 1.21.6

- Source: https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.6.tar.gz
- In: libcppZarr, all platforms.
- License: BSD-3-Clause.
- Built in with it, from its `internal-complibs/`: LZ4 1.9.4 (BSD-2-Clause), Zstandard 1.5.6 (BSD-3-Clause, of its BSD-3-Clause OR GPL-2.0-only) and zlib 1.3.1 (Zlib); and its BloscLZ (derived from FastLZ, MIT) and bitshuffle (MIT) code. Snappy is not built; `STDINT.txt` is for MSVC builds, which cpp-zarr does not make.
- Files: `licenses/c-blosc-1.21.6/LICENSE.txt`; `licenses/c-blosc-1.21.6/LICENSES/` (`LZ4.txt`, `ZLIB.txt`, `FASTLZ.txt`, `BITSHUFFLE.txt`, and the folder's other files: it is kept whole); `licenses/zstd-1.5.6/LICENSE`, from https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz (c-blosc's `LICENSES/` has no Zstandard file).

## c-blosc2 2.23.1

- Source: https://github.com/Blosc/c-blosc2/archive/refs/tags/v2.23.1.tar.gz
- In: libcppZarr, all platforms.
- License: BSD-3-Clause.
- Built in with it: from its `internal-complibs/`, LZ4 1.10.0 (BSD-2-Clause), Zstandard 1.5.7 (BSD-3-Clause, of its BSD-3-Clause OR GPL-2.0-only) and zlib-ng 2.0.7 (Zlib); its BloscLZ (FastLZ, MIT) and bitshuffle (MIT) code; and its plugins, the Blosc team's (BSD-3-Clause) but for two works they carry: the zfp 0.5.5 codec (BSD-3-Clause) and, in the ndlz codec, xxHash 0.8.2 (BSD-2-Clause).
- Files: `licenses/c-blosc2-2.23.1/LICENSE.txt`; `licenses/c-blosc2-2.23.1/LICENSES/` (`LZ4.txt`, `ZSTD.txt`, `ZLIB.txt` (zlib-ng's), `FASTLZ.txt`, `BITSHUFFLE.txt`; kept whole); `licenses/zfp-0.5.5/LICENSE`, from https://github.com/LLNL/zfp/releases/download/0.5.5/zfp-0.5.5.tar.gz, and `licenses/xxHash-0.8.2/LICENSE`, from https://github.com/Cyan4973/xxHash/archive/refs/tags/v0.8.2.tar.gz (c-blosc2 carries their sources, with their copyright lines, but not these files).

## zlib 1.3.1

- Source: https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
- In: libcppZarr, all platforms.
- License: Zlib.
- File: `licenses/zlib-1.3.1/LICENSE`.

## nlohmann/json 3.12.0

- Source: https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
- In: libcppZarr, all platforms (header-only, compiled in).
- License: MIT (the parts of its headers by other authors are MIT too, by their SPDX tags).
- File: `licenses/nlohmann_json-3.12.0/LICENSE.MIT`.

## libuuid 1.0.3

- Source: `dependencies/libuuid-1.0.3.tar.gz` in this repository (the standalone libuuid of util-linux).
- In: libcppZarr on Linux and macOS: the Linux and macOS wheels, `linux/` and `macArm/`.
- License: BSD-3-Clause ("Modified BSD"). Copyright (C) 1996-1999, 2007 Theodore Ts'o.
- Files: `licenses/libuuid-1.0.3/COPYING`, and `licenses/libuuid-1.0.3/gen_uuid.c-header.txt`, the license terms: the first 33 lines of its `gen_uuid.c`, unchanged (the tarball does not include the license file its `COPYING` names, and its sources carry the terms in their headers).

## pybind11

- Source: https://pypi.org/project/pybind11/ (a build takes the newest that `pybind11>=2.10.0` allows; the file is pybind11 3.1.0's, the version the 1.8.0 wheels are built with).
- In: the module (its headers are compiled in), every wheel.
- License: BSD-3-Clause.
- File: `licenses/pybind11/LICENSE`.

## The GCC runtime libraries

- Source: https://gcc.gnu.org (libstdc++, libgcc and libgomp).
- In: libstdc++ linked statically into libcppZarr and the module (all platforms) and the Linux mex files; libgcc linked statically into libcppZarr on macOS and Windows. In the Linux wheels these are gcc-toolset-14's (GCC 14.2.1). Bundled: GCC's OpenMP runtime libgomp in every wheel and MATLAB folder (Linux: the manylinux_2_28 image's, from AlmaLinux 8's GCC 8.5.0 libgomp package; macOS: Homebrew's GCC 13; Windows: MinGW-w64's GCC 12.2); libgcc_s in the Windows wheels and `windows/` (`libgcc_s_seh-1.dll`) and in `macArm/` (`libgcc_s.1.1.0.dylib`); libstdc++ in `macArm/` (`libstdc++.6.0.32.dylib`).
- License: GPL-3.0-or-later WITH GCC-exception-3.1. The GCC Runtime Library Exception lets programs compiled with GCC and linked with these libraries be distributed under licenses of their own (https://www.gnu.org/licenses/gcc-exception-3.1.html).
- Files: `licenses/gcc-runtime/COPYING3` and `licenses/gcc-runtime/COPYING.RUNTIME` (the same in GCC 12.2, 13.2 and 14.2).

## mingw-w64 winpthreads

- Source: the MinGW-w64 toolchain of the Windows builds (GCC 12.2), https://www.mingw-w64.org
- In: the Windows wheels and `windows/` (`libwinpthread-1.dll`, bundled).
- License: MIT, with parts under BSD-3-Clause (Lockless Inc.).
- File: `licenses/mingw-w64-winpthreads/COPYING`.

## mingw-w64 runtime

- Source: the same MinGW-w64 toolchain.
- In: libcppZarr, the module and the mex files on Windows (its startup code, support library and import libraries are linked statically into every binary MinGW-w64 builds).
- License: the mingw-w64 project's BSD-style license, with the notices of its parts under other terms, as the file lists them.
- File: `licenses/mingw-w64-runtime/COPYING.MinGW-w64-runtime.txt`.
