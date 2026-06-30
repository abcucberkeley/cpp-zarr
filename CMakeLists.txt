cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

project(
  cpp-zarr
  VERSION 1.5.0
  LANGUAGES CXX)

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

    # Build the UUID library using Autotools
    ExternalProject_Add(uuid_external
        SOURCE_DIR ${UUID_SOURCE_DIR}
        CONFIGURE_COMMAND ${UUID_SOURCE_DIR}/configure --with-pic
        BUILD_COMMAND make
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
    )

    # Specify the include directories and link the UUID library
    ExternalProject_Get_Property(uuid_external source_dir)
    ExternalProject_Get_Property(uuid_external binary_dir)
endif()

# ---- Fetched dependencies -------------------------------------------------

# Options consumed by c-blosc / c-blosc2 (must be set before they configure).
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# nlohmann/json: don't build its test suite
set(JSON_BuildTests OFF CACHE INTERNAL "")

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
set(ZLIB_SRCS
    adler32.c compress.c crc32.c deflate.c gzclose.c gzlib.c gzread.c gzwrite.c
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
    target_include_directories(cppZarr_shared PRIVATE ${source_dir}/include)
    target_link_libraries(cppZarr_shared PRIVATE ${binary_dir}/.libs/libuuid.a)
endif()

# blosc/blosc2 propagate their include dirs through their targets, but stay explicit.
target_include_directories(cppZarr_shared PRIVATE
    ${cblosc_SOURCE_DIR}/blosc
    ${cblosc2_SOURCE_DIR}/include
)

# Link against the fetched static libraries by target name.
target_link_libraries(cppZarr_shared PRIVATE
    #"-Wl,--allow-multiple-definition"   # enable if standalone zlib & blosc's bundled zlib clash at link
    blosc2_static
    blosc_static
    zlibstaticCPPZARR
    nlohmann_json::nlohmann_json
    -static-libgcc
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

install(TARGETS cppZarr_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install JSON headers (zarr.h includes <nlohmann/json.hpp>)
install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/include/nlohmann
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
