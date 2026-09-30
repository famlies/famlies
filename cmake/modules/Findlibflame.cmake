# Findlibflame
# ------------
#
# Provides libflame (https://github.com/flame/libflame) as the imported target
# libflame::libflame. There is no CMake package config to find, so this module
# downloads and builds libflame itself: the commit that
# external/versions.cmake pins as FAMLIES_TRACKED_LIBFLAME_*, which must be
# included first.
#
# libflame calls BLAS, which it gets from BLIS::BLIS, so that must exist first
# and have BLIS's BLAS layer.

if(TARGET libflame::libflame)
    return()
endif()

include(ExternalProject)
include(FetchContent)
include(FindPackageHandleStandardArgs)

# libflame has no CMake build, so this only fetches it.
FetchContent_Declare(libflame
    URL "${FAMLIES_TRACKED_LIBFLAME_URL}"
    URL_HASH SHA256=${FAMLIES_TRACKED_LIBFLAME_SHA256}
    DOWNLOAD_NO_PROGRESS TRUE)
FetchContent_MakeAvailable(libflame)

set(LIBFLAME_PREFIX "${FETCHCONTENT_BASE_DIR}/libflame-install")
set(LIBFLAME_INCLUDE_DIR "${LIBFLAME_PREFIX}/include")
set(LIBFLAME_LIBRARY
    "${LIBFLAME_PREFIX}/lib/\
${CMAKE_STATIC_LIBRARY_PREFIX}flame${CMAKE_STATIC_LIBRARY_SUFFIX}")

find_package_handle_standard_args(libflame REQUIRED_VARS LIBFLAME_LIBRARY)

if(libflame_FOUND)
    find_program(LIBFLAME_MAKE_EXECUTABLE NAMES gmake make REQUIRED
                 DOC "The make executable used to build libflame")
    cmake_host_system_information(RESULT _libflame_jobs
                                  QUERY NUMBER_OF_PHYSICAL_CORES)

    # libflame builds in its source tree, which may be shared and read-only
    # (FETCHCONTENT_SOURCE_DIR_LIBFLAME), so a copy is used. No Fortran
    # probes: nothing needs Fortran, and BLIS's BLAS symbols have the default
    # trailing underscore. -fno-common (not Apple clang's default) stops ld64
    # warning that fla_error_string's guessed alignment exceeds the segment's.
    ExternalProject_Add(libflame_external
        DOWNLOAD_COMMAND    ${CMAKE_COMMAND} -E copy_directory
                            "${libflame_SOURCE_DIR}" <SOURCE_DIR>
        UPDATE_COMMAND      ""
        PREFIX              "${FETCHCONTENT_BASE_DIR}/libflame-ep"
        SOURCE_DIR          "${FETCHCONTENT_BASE_DIR}/libflame-src-copy"
        INSTALL_DIR         "${LIBFLAME_PREFIX}"
        BUILD_IN_SOURCE     TRUE
        CONFIGURE_COMMAND   <SOURCE_DIR>/configure
                            --prefix=<INSTALL_DIR>
                            --disable-autodetect-f77-ldflags
                            --disable-autodetect-f77-name-mangling
                            CC=${CMAKE_C_COMPILER}
                            "CFLAGS=-O3 -fno-common"
        BUILD_COMMAND       ${LIBFLAME_MAKE_EXECUTABLE} -j${_libflame_jobs}
        INSTALL_COMMAND     ${LIBFLAME_MAKE_EXECUTABLE} install
        INSTALL_BYPRODUCTS  "${LIBFLAME_LIBRARY}"
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        LOG_OUTPUT_ON_FAILURE TRUE)

    # An imported include directory must exist before generation.
    file(MAKE_DIRECTORY "${LIBFLAME_INCLUDE_DIR}")

    add_library(libflame::libflame STATIC IMPORTED GLOBAL)
    set_target_properties(libflame::libflame PROPERTIES
        IMPORTED_LOCATION             "${LIBFLAME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBFLAME_INCLUDE_DIR}")
    target_link_libraries(libflame::libflame INTERFACE BLIS::BLIS)
    # Otherwise nothing triggers the build.
    add_dependencies(libflame::libflame libflame_external)
endif()

mark_as_advanced(LIBFLAME_MAKE_EXECUTABLE LIBFLAME_INCLUDE_DIR
                 LIBFLAME_LIBRARY)
