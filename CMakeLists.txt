cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# cpp-zarr's optimization flags for each build type, unless given on the command
# line (-DCMAKE_CXX_FLAGS_RELEASE=...). (Cache entries made before project(), which
# keeps them instead of setting its own defaults.)
if(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG "-g" CACHE STRING "Flags used by the C++ compiler for Debug builds")
endif()
if(NOT DEFINED CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG" CACHE STRING "Flags used by the C++ compiler for RelWithDebInfo builds")
endif()
if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Flags used by the C++ compiler for Release builds")
endif()

project(
  cpp-zarr
  VERSION 1.7.2
  LANGUAGES CXX)

# Warnings, added to the flags given on the command line or in CXXFLAGS
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")

include(GNUInstallDirs)
include(FetchContent)

# Build the fetched static deps with position-independent code so they can be
# linked into our shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the RPATH to $ORIGIN when installing if applicable
set(CMAKE_INSTALL_RPATH $ORIGIN)

# OpenMP is required for C++ only. (Requiring it for C too would break toolchains
# whose C compiler lacks OpenMP -- e.g. Apple Clang on macOS -- and the C sources
# we compile below, zlib, do not use OpenMP.)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if (UNIX)
    execute_process(
        COMMAND rm -rf libuuid-1.0.3
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
    )
	execute_process(
		COMMAND tar -xf libuuid-1.0.3.tar.gz
    	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
	)

	include(ExternalProject)

    # Set the path to your Makefile.am
    set(UUID_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libuuid-1.0.3/")

    # Build the UUID library using Autotools. libuuid 1.0.3 is old C that calls
    # flock() (and similar) without their declaring headers; GCC 14 and recent
    # Clang promote implicit function declarations to hard errors by default, so
    # relax those diagnostics to keep this dependency building.
    ExternalProject_Add(uuid_external
        SOURCE_DIR ${UUID_SOURCE_DIR}
        CONFIGURE_COMMAND ${UUID_SOURCE_DIR}/configure --with-pic "CFLAGS=-O2 -fPIC -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion"
        BUILD_COMMAND make
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
    )

    # Specify the include directories and link the UUID library
    ExternalProject_Get_Property(uuid_external source_dir)
    ExternalProject_Get_Property(uuid_external binary_dir)
    # The sources include <uuid/uuid.h>: the bundled library's header, which its
    # tarball keeps at its top level, goes in a uuid/ folder of the build, so
    # systems without libuuid's headers (uuid-dev) build too, and the header always
    # matches the library linked in
    set(UUID_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/uuid_include")
    configure_file(${UUID_SOURCE_DIR}/uuid.h ${UUID_INCLUDE_DIR}/uuid/uuid.h COPYONLY)
endif()

# ---- Fetched dependencies -------------------------------------------------

# Options consumed by c-blosc / c-blosc2 (must be set before they configure).
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# The reader also calls zstd itself (Zarr v3's zstd codec), using the zstd bundled
# with c-blosc2 that its library already contains
set(DEACTIVATE_ZSTD OFF CACHE BOOL "" FORCE)
set(PREFER_EXTERNAL_ZSTD OFF CACHE BOOL "" FORCE)
# nlohmann/json: don't build its test suite
set(JSON_BuildTests OFF CACHE INTERNAL "")
# c-blosc still declares cmake_minimum_required(VERSION 2.8.12), which CMake 4
# rejects (it dropped compatibility with versions before 3.5). This lets CMake 4
# configure it with the 3.5 policies; older CMake versions ignore it.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(cblosc
    URL      https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.6.tar.gz
    URL_HASH SHA256=9fcd60301aae28f97f1301b735f966cc19e7c49b6b4321b839b4579a0c156f38)
FetchContent_Declare(cblosc2
    URL      https://github.com/Blosc/c-blosc2/archive/refs/tags/v2.23.1.tar.gz
    URL_HASH SHA256=3a1a55d1e3794fb2b51a12e722d611b3e577443abb7ff9951666511f576ea3da)
FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
    URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187)
FetchContent_Declare(zlib
    URL      https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23)

FetchContent_MakeAvailable(cblosc cblosc2 nlohmann_json)

# c-blosc/c-blosc2 build their own bundled zlib/zlib-ng as CMake targets named
# "zlib"/"zlibstatic", so stock zlib cannot be allowed to create those same
# targets. Populate zlib's sources only and compile them into a uniquely-named
# static library (zlib ships no prebuilt zconf.h, so generate it from zconf.h.in).
FetchContent_GetProperties(zlib)
if(NOT zlib_POPULATED)
    FetchContent_Populate(zlib)
endif()
set(ZLIB_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/zlib_gen")
configure_file(${zlib_SOURCE_DIR}/zconf.h.in ${ZLIB_GEN_DIR}/zconf.h COPYONLY)
# cpp-zarr only uses zlib's in-memory inflate/deflate streaming API, so we omit
# the gzFile file-API sources (gz*.c). They call POSIX lseek/read/write/close
# and, when zlib is built outside its own configure step (which defines
# Z_HAVE_UNISTD_H), they don't pull in <unistd.h> -- which Apple Clang rejects
# as implicit declarations (hard error), breaking the macOS build.
set(ZLIB_SRCS
    adler32.c compress.c crc32.c deflate.c
    infback.c inffast.c inflate.c inftrees.c trees.c uncompr.c zutil.c)
list(TRANSFORM ZLIB_SRCS PREPEND "${zlib_SOURCE_DIR}/")
# Enable C only here (for zlib's C sources), after the OpenMP check, so the
# project stays C++ and OpenMP is never required for the C compiler.
enable_language(C)
add_library(zlibstaticCPPZARR STATIC ${ZLIB_SRCS})
target_include_directories(zlibstaticCPPZARR PUBLIC ${zlib_SOURCE_DIR} ${ZLIB_GEN_DIR})

# ---- cpp-zarr shared library ----------------------------------------------

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(cppZarr_shared SHARED ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp)
set_target_properties(cppZarr_shared PROPERTIES OUTPUT_NAME cppZarr)
set_target_properties(cppZarr_shared PROPERTIES PUBLIC_HEADER "${SRC_DIR}/zarr.h;${SRC_DIR}/helperfunctions.h;${SRC_DIR}/parallelreadzarr.h;${SRC_DIR}/parallelwritezarr.h")

if(UNIX)
    add_dependencies(cppZarr_shared uuid_external)
    target_include_directories(cppZarr_shared BEFORE PRIVATE ${UUID_INCLUDE_DIR})
    target_link_libraries(cppZarr_shared PRIVATE ${binary_dir}/.libs/libuuid.a)
endif()

# blosc/blosc2 propagate their include dirs through their targets, but stay explicit.
# zstd.h comes from the zstd bundled with c-blosc2.
file(GLOB CPPZARR_ZSTD_DIR LIST_DIRECTORIES true "${cblosc2_SOURCE_DIR}/internal-complibs/zstd-*")
target_include_directories(cppZarr_shared PRIVATE
    ${cblosc_SOURCE_DIR}/blosc
    ${cblosc2_SOURCE_DIR}/include
    ${CPPZARR_ZSTD_DIR}
)

# Link against the fetched static libraries by target name.
target_link_libraries(cppZarr_shared PRIVATE
    #"-Wl,--allow-multiple-definition"   # enable if standalone zlib & blosc's bundled zlib clash at link
    blosc2_static
    blosc_static
    zlibstaticCPPZARR
    nlohmann_json::nlohmann_json
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

if(MINGW)
    target_link_options(cppZarr_shared PRIVATE -Wl,--allow-multiple-definition)
endif()
# CPPZARR_PRIVATE_RUNTIME (Linux): keep the static C++ runtime (and the other static libraries)
# private to the library. Exported, programs that load another C++ runtime after cpp-zarr (Python
# with PyTorch, for one) bound some of its symbols to this copy and crashed, so the Python wheel
# turns it on. Off by default: programs built against the library's exported runtime (Parallel
# Imaris Writer) need it.
#
# On Linux the library's own calls are bound to its own static blosc, zstd, LZ4, zlib and C++
# runtime (-Bsymbolic; the symbols stay exported). Otherwise, once the host has loaded another
# zstd into its global scope (MATLAB loads its own at startup), the library's zstd calls went to
# that one: writes of zstd arrays from MATLAB took 1.25 times as long. The unwinder is the shared
# libgcc_s every C++ program uses: a private static copy could not unwind the library's
# exceptions through its callers' code. (macOS and Windows already bind a library's calls to its
# own definitions.)
option(CPPZARR_PRIVATE_RUNTIME "Keep the library's static C++ runtime private (Linux)" OFF)
if(UNIX AND NOT APPLE)
    if(CPPZARR_PRIVATE_RUNTIME)
        target_link_options(cppZarr_shared PRIVATE -Wl,--exclude-libs,ALL)
    endif()
    target_link_options(cppZarr_shared PRIVATE -Wl,-Bsymbolic)
else()
    target_link_libraries(cppZarr_shared PRIVATE -static-libgcc)
endif()

install(TARGETS cppZarr_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install JSON headers (zarr.h includes <nlohmann/json.hpp>)
install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/include/nlohmann
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ---- Round-trip smoke tests (write random data, read it back, verify) ------
# Enable with -DCPPZARR_BUILD_TESTS=ON, then run `ctest` in the build dir.
# (BUILD_TESTS is forced OFF above for the bundled blosc deps, so this uses a
# project-specific option name to avoid clashing with it.)
option(CPPZARR_BUILD_TESTS "Build and register the round-trip smoke tests" OFF)
if(CPPZARR_BUILD_TESTS)
    enable_testing()
    add_executable(roundtripTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.cpp)
    set_target_properties(roundtripTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(roundtripTest PRIVATE ${SRC_DIR} ${nlohmann_json_SOURCE_DIR}/include)
    # Arrays written by zarr-python and TensorStore (made by tests/make_test_arrays.py)
    target_compile_definitions(roundtripTest PRIVATE CPPZARR_TEST_ARRAYS="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_arrays")
    target_link_libraries(roundtripTest PRIVATE cppZarr_shared OpenMP::OpenMP_CXX)
    add_test(NAME roundtrip COMMAND roundtripTest ${CMAKE_CURRENT_BINARY_DIR})
endif()
