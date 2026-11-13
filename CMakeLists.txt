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
  VERSION 1.7.3
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
# and only that: entries the toolchain itself adds when linking (a Spack-built GCC adds its own lib64) are left
# out of installed libraries (CMake 3.16 and later), which find the GCC runtime next to them, as MATLAB's
# release folders ship it, or as the system has it (libcppZarr needs nothing newer than GCC 6's)
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)

# No absolute path of the machine that builds it in the library: __FILE__ (c-blosc2's error messages use it)
# and debug information name the source files relative to the source and build folders, and to dependencies'
# sources given with FETCHCONTENT_SOURCE_DIR_* (for every target, the dependencies' too, so set before they
# are added). With GCC the last option that matches wins, so the more specific folders come later.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffile-prefix-map=${CMAKE_SOURCE_DIR}=." CPPZARR_HAS_FILE_PREFIX_MAP)
if(CPPZARR_HAS_FILE_PREFIX_MAP)
    set(CPPZARR_PREFIX_MAPS "${CMAKE_SOURCE_DIR}=." "${CMAKE_BINARY_DIR}=.")
    foreach(dep CBLOSC CBLOSC2 NLOHMANN_JSON ZLIB)
        if(FETCHCONTENT_SOURCE_DIR_${dep})
            get_filename_component(depDir "${FETCHCONTENT_SOURCE_DIR_${dep}}" ABSOLUTE)
            string(TOLOWER ${dep} depName)
            list(APPEND CPPZARR_PREFIX_MAPS "${depDir}=./_deps/${depName}-src")
        endif()
    endforeach()
    foreach(map ${CPPZARR_PREFIX_MAPS})
        add_compile_options("-ffile-prefix-map=${map}")
    endforeach()
endif()

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

# c-blosc compiles its own pthreads into blosc.c for every Windows build (win32/pthread.c), under
# pthread_create's and pthread_cond_*'s own names. Linked into the library together with MinGW's C++
# runtime (and --allow-multiple-definition, below), those took the place of winpthreads' for the
# runtime too: its threads, condition variables and thread-safe statics then went to blosc's, which
# do not know winpthreads' types, and the first static with a run-time initializer crashed every
# read. Renamed, blosc keeps its own and the runtime gets winpthreads'.
if(WIN32)
    target_compile_definitions(blosc_static PRIVATE
        pthread_create=blosc_win32_pthread_create
        pthread_cond_init=blosc_win32_pthread_cond_init
        pthread_cond_destroy=blosc_win32_pthread_cond_destroy
        pthread_cond_wait=blosc_win32_pthread_cond_wait
        pthread_cond_signal=blosc_win32_pthread_cond_signal
        pthread_cond_broadcast=blosc_win32_pthread_cond_broadcast)
endif()

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

# (GPU reads and writes: the CUDA driver and nvCOMP are loaded at run time, so they need nothing to build)
add_library(cppZarr_shared SHARED ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp
            ${SRC_DIR}/gpureadzarr.cpp ${SRC_DIR}/gpuwritezarr.cpp ${SRC_DIR}/gpu/cudaapi.cpp ${SRC_DIR}/gpu/gpucommon.cpp)
set_target_properties(cppZarr_shared PROPERTIES OUTPUT_NAME cppZarr)
set_target_properties(cppZarr_shared PROPERTIES PUBLIC_HEADER "${SRC_DIR}/zarr.h;${SRC_DIR}/helperfunctions.h;${SRC_DIR}/parallelreadzarr.h;${SRC_DIR}/parallelwritezarr.h;${SRC_DIR}/gpureadzarr.h;${SRC_DIR}/gpuwritezarr.h")

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
    ${CMAKE_DL_LIBS}
)

if(MINGW)
    target_link_options(cppZarr_shared PRIVATE -Wl,--allow-multiple-definition)
endif()
# CPPZARR_PRIVATE_RUNTIME (Linux and macOS, on by default, as cpp-tiff's CPPTIFF_PRIVATE_RUNTIME is): keep the
# library's static C++ runtime, and the nlohmann/json code it builds in, out of its exports. Exported, programs and
# modules with a C++ runtime of their own bound some of its symbols to this copy, two runtimes in one process: Python
# crashed when PyTorch was imported after cpp-zarr had been used (Linux), std::filesystem::copy aborted in a program
# with Homebrew's libstdc++ (macOS), and on macOS the loader merges the exported weak definitions of one name across
# every image a process loads (the first one loaded wins), so that with cpp-tiff's Python module loaded first,
# cpp-zarr's first write crashed, running the other module's copy of the runtime's template code on that copy's
# state. The library exports its API, the zarr class and the functions of its installed headers, and the programs,
# modules and MEX files that link it bring their own C++ runtime. Turn it off for a program built to take its runtime
# from the library's exports (as Parallel Imaris Writer's releases before its own runtime were), and in a build that
# gives this target a version script of its own: GNU ld takes one anonymous version script ("anonymous version tag
# cannot be combined with other version tags"). Linux: no symbol of the static libraries exported (--exclude-libs:
# the C++ runtime, blosc, zstd, LZ4, zlib and libuuid, none of whose headers the library installs), and a version
# script for the template code of std and nlohmann/json that the library's own objects instantiate (it keeps default
# visibility); macOS: an unexported-symbols list. The patterns are mangled names: std's, its short forms (Sa, Sb, Ss,
# Si, So, Sd), its typeinfo, vtables, guards, thunks and TLS functions, __gnu_cxx's, __cxxabiv1's, the typeinfo of
# built-in types, operator new and delete and the C++ ABI's C functions (cpp-tiff's list), thunks to the short forms'
# members (on macOS the library exported eight, of std::basic_iostream's, basic_istream's and basic_ostream's
# destructors: _ZThn16_NSdD1Ev and the like), and nlohmann's.
#
# On Linux the library's own calls are bound to its own static blosc, zstd, LZ4, zlib and C++
# runtime (-Bsymbolic; with CPPZARR_PRIVATE_RUNTIME off, those symbols stay exported). Otherwise,
# once the host has loaded another zstd into its global scope (MATLAB loads its own at startup), the
# library's zstd calls went to that one: writes of zstd arrays from MATLAB took 1.25 times as long.
# The unwinder is the shared libgcc_s every C++ program uses: a private static copy could not unwind
# the library's exceptions through its callers' code. (macOS and Windows already bind a library's
# calls to its own definitions, all but the weak ones on macOS.)
option(CPPZARR_PRIVATE_RUNTIME "Keep the library's static C++ runtime private (Linux and macOS)" ON)
set(CPPZARR_RUNTIME_SYMBOLS
    "_ZNSt*" "_ZNKSt*" "_ZSt*" "_ZTISt*" "_ZTSSt*" "_ZTVSt*" "_ZTTSt*" "_ZGVZNSt*" "_ZGVNSt*" "_ZGVZNKSt*"
    "_ZTINSt*" "_ZTSNSt*" "_ZTVNSt*" "_ZTTNSt*" "_ZTHSt*" "_ZTWSt*" "_ZTHNSt*" "_ZTWNSt*"
    "_ZNSa*" "_ZNKSa*" "_ZNSb*" "_ZNKSb*" "_ZNSs*" "_ZNKSs*" "_ZNSi*" "_ZNKSi*" "_ZNSo*" "_ZNKSo*" "_ZNSd*" "_ZNKSd*"
    "_ZTISa*" "_ZTISb*" "_ZTISs*" "_ZTISi*" "_ZTISo*" "_ZTISd*" "_ZTSSa*" "_ZTSSb*" "_ZTSSs*" "_ZTSSi*" "_ZTSSo*" "_ZTSSd*"
    "_ZTVSa*" "_ZTVSb*" "_ZTVSs*" "_ZTVSi*" "_ZTVSo*" "_ZTVSd*" "_ZTTSi*" "_ZTTSo*" "_ZTTSd*"
    "_ZThn*St*" "_ZTv*St*" "_ZTch*St*" "_ZTh*_NS[abdios]*" "_ZTv*_NS[abdios]*" "_ZTc*_NS[abdios]*"
    "_ZN9__gnu_cxx*" "_ZNK9__gnu_cxx*" "_ZTIN9__gnu_cxx*" "_ZTSN9__gnu_cxx*" "_ZTVN9__gnu_cxx*" "_ZGVZN9__gnu_cxx*"
    "_ZN10__cxxabiv1*" "_ZNK10__cxxabiv1*" "_ZTIN10__cxxabiv1*" "_ZTSN10__cxxabiv1*" "_ZTVN10__cxxabiv1*"
    "_ZTI?" "_ZTS?" "_ZTIP?" "_ZTSP?" "_ZTIPK?" "_ZTSPK?" "_ZTID?" "_ZTSD?" "_ZTIPD?" "_ZTSPD?" "_ZTIPKD?" "_ZTSPKD?"
    "_ZZNSt*" "_ZZNKSt*" "__cxa_*" "__gxx_personality*" "__dynamic_cast" "_Znw*" "_Zna*" "_Zdl*" "_Zda*"
    "_ZN8nlohmann*" "_ZNK8nlohmann*" "_ZZN8nlohmann*" "_ZZNK8nlohmann*" "_ZGVZN8nlohmann*"
    "_ZTIN8nlohmann*" "_ZTSN8nlohmann*" "_ZTVN8nlohmann*")
if(UNIX AND NOT APPLE)
    if(CPPZARR_PRIVATE_RUNTIME)
        string(REPLACE ";" ";\n    " CPPZARR_RUNTIME_LOCAL "${CPPZARR_RUNTIME_SYMBOLS}")
        set(CPPZARR_RUNTIME_MAP "${CMAKE_CURRENT_BINARY_DIR}/cppzarr_private_runtime.map")
        file(WRITE "${CPPZARR_RUNTIME_MAP}" "{\n  local:\n    ${CPPZARR_RUNTIME_LOCAL};\n};\n")
        target_link_options(cppZarr_shared PRIVATE -Wl,--exclude-libs,ALL "LINKER:--version-script,${CPPZARR_RUNTIME_MAP}")
        set_property(TARGET cppZarr_shared APPEND PROPERTY LINK_DEPENDS "${CPPZARR_RUNTIME_MAP}")
    endif()
    target_link_options(cppZarr_shared PRIVATE -Wl,-Bsymbolic)
else()
    target_link_libraries(cppZarr_shared PRIVATE -static-libgcc)
    if(APPLE AND CPPZARR_PRIVATE_RUNTIME)
        # (Mach-O names start with one more _)
        list(TRANSFORM CPPZARR_RUNTIME_SYMBOLS PREPEND "_" OUTPUT_VARIABLE CPPZARR_RUNTIME_MACHO)
        string(REPLACE ";" "\n" CPPZARR_RUNTIME_TEXT "${CPPZARR_RUNTIME_MACHO}")
        set(CPPZARR_RUNTIME_LIST "${CMAKE_CURRENT_BINARY_DIR}/cppzarr_private_runtime.txt")
        file(WRITE "${CPPZARR_RUNTIME_LIST}" "${CPPZARR_RUNTIME_TEXT}\n")
        target_link_options(cppZarr_shared PRIVATE "LINKER:-unexported_symbols_list,${CPPZARR_RUNTIME_LIST}")
        set_property(TARGET cppZarr_shared APPEND PROPERTY LINK_DEPENDS "${CPPZARR_RUNTIME_LIST}")
    endif()
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
    # GPU reads against the CPU reader (skipped without an NVIDIA GPU, driver and nvCOMP)
    add_executable(gpuTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/gputest.cpp)
    set_target_properties(gpuTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(gpuTest PRIVATE ${SRC_DIR} ${nlohmann_json_SOURCE_DIR}/include)
    target_compile_definitions(gpuTest PRIVATE CPPZARR_TEST_ARRAYS="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_arrays")
    target_link_libraries(gpuTest PRIVATE cppZarr_shared OpenMP::OpenMP_CXX)
    add_test(NAME gpu COMMAND gpuTest ${CMAKE_CURRENT_BINARY_DIR})
    # GPU writes against CPU writes (skipped without an NVIDIA GPU, driver and nvCOMP)
    add_executable(gpuWriteTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpuwritetest.cpp)
    set_target_properties(gpuWriteTest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(gpuWriteTest PRIVATE ${SRC_DIR} ${nlohmann_json_SOURCE_DIR}/include)
    target_compile_definitions(gpuWriteTest PRIVATE CPPZARR_TEST_ARRAYS="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_arrays")
    target_link_libraries(gpuWriteTest PRIVATE cppZarr_shared OpenMP::OpenMP_CXX)
    add_test(NAME gpuwrite COMMAND gpuWriteTest ${CMAKE_CURRENT_BINARY_DIR})
endif()
