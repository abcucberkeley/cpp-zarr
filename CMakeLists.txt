cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

project(
  cpp-zarr
  VERSION 1.1.1
  LANGUAGES CXX)

# OpenMP is Required
find_package(OpenMP REQUIRED)

if (UNIX)
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
    #target_include_directories(cppZarr_shared PRIVATE ${source_dir}/include)
    #target_link_libraries(cppZarr_shared PRIVATE ${binary_dir}/.libs/libuuid.a)
endif()


# Add c-blosc
set(CBLOSC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/c-blosc-1.21.0")
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CBLOSC_DIR})
set_property(TARGET blosc_static PROPERTY POSITION_INDEPENDENT_CODE ON)


set(CBLOSC2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/c-blosc2-2.8.0")
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CBLOSC2_DIR})
set_property(TARGET blosc2_static PROPERTY POSITION_INDEPENDENT_CODE ON)

set(JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/json-3.11.2")
add_subdirectory(${JSON_DIR})

set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
set(ZLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib-1.2.8")
add_subdirectory(${ZLIB_DIR})
set_property(TARGET zlibstaticCPPZARR PROPERTY POSITION_INDEPENDENT_CODE ON)


set(CBLOSC_INCLUDE_DIR "${CBLOSC_DIR}/blosc")
set(CBLOSC2_INCLUDE_DIR "${CBLOSC2_DIR}/include")
set(JSON_INCLUDE_DIR "${JSON_DIR}/include")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(cppZarr_shared SHARED ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp)
set_target_properties(cppZarr_shared PROPERTIES OUTPUT_NAME cppZarr)
add_dependencies(cppZarr_shared blosc_static)
add_dependencies(cppZarr_shared blosc2_static)
add_dependencies(cppZarr_shared zlibstaticCPPZARR)
add_dependencies(cppZarr_shared uuid_external)

#add_library(cppZarrStatic STATIC ${SRC_DIR}/zarr.cpp ${SRC_DIR}/helperfunctions.cpp ${SRC_DIR}/parallelreadzarr.cpp ${SRC_DIR}/parallelwritezarr.cpp)
#set_property(TARGET cppZarr_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(cppZarr_shared PROPERTIES PUBLIC_HEADER "${SRC_DIR}/zarr.h;${SRC_DIR}/helperfunctions.h;${SRC_DIR}/parallelreadzarr.h;${SRC_DIR}/parallelwritezarr.h")

if(UNIX)
    target_include_directories(cppZarr_shared PRIVATE ${source_dir}/include)
    target_link_libraries(cppZarr_shared PRIVATE ${binary_dir}/.libs/libuuid.a)
endif()

#message(NOTICE "${CBLOSC_INCLUDE_DIR}")
#message(NOTICE "${CBLOSC2_INCLUDE_DIR}")
#message(NOTICE "${JSON_INCLUDE_DIR}")
#message(NOTICE "${ZLIB_DIR}")

# Include the header files from blosc, blosc2, and json
target_include_directories(cppZarr_shared PRIVATE
    ${CBLOSC_INCLUDE_DIR}
    ${CBLOSC2_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
    ${ZLIB_DIR}
)


#target_include_directories(cppZarrStatic PRIVATE
#    ${CBLOSC_INCLUDE_DIR}
#    ${CBLOSC2_INCLUDE_DIR}
#    ${JSON_INCLUDE_DIR}
#    ${ZLIB_DIR}
#)

# Link against the blosc and blosc2 libraries
target_link_libraries(cppZarr_shared PRIVATE
    #"-Wl,--allow-multiple-definition"  
    ${CMAKE_BINARY_DIR}/dependencies/c-blosc2-2.8.0/blosc/libblosc2.a
    ${CMAKE_BINARY_DIR}/dependencies/c-blosc-1.21.0/blosc/libblosc.a
    ${CMAKE_BINARY_DIR}/dependencies/zlib-1.2.8/libz.a
    -static-libgcc
    -static-libstdc++
    OpenMP::OpenMP_CXX
)

install(TARGETS cppZarr_shared
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
