# FindBLIS
# --------
#
# Provides BLIS (https://github.com/flame/blis) as the imported target
# BLIS::BLIS. There is no CMake package config to find, so this module
# downloads and builds BLIS itself: the release that external/versions.cmake
# pins as FAMLIES_TRACKED_BLIS_RELEASE_*, which must be included first.

if(CMAKE_SCRIPT_MODE_FILE)
    set(jobs "")
    if("$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" MATCHES "^[1-9][0-9]*$")
        set(jobs "-j${CMAKE_MATCH_0}")
    endif()
    execute_process(COMMAND "${MAKE_EXE}" ${jobs} ${TARGET}
                    WORKING_DIRECTORY "${WORKDIR}"
                    RESULT_VARIABLE result)
    if(result)
        message(FATAL_ERROR
                "BLIS: '${MAKE_EXE} ${jobs} ${TARGET}' failed (${result})")
    endif()
    return()
endif()

if(TARGET BLIS::BLIS)
    return()
endif()

include(ExternalProject)
include(FetchContent)
include(FindPackageHandleStandardArgs)

set(BLIS_CONFNAME "auto"
    CACHE STRING "BLIS sub-configuration (confname passed to configure)")
set(BLIS_CONFIGURE_OPTIONS ""
    CACHE STRING "Options passed verbatim to BLIS's configure (see its --help)")
# Options the module sets itself. Everything else in BLIS_CONFIGURE_OPTIONS
# goes through unchecked, on purpose.
foreach(_blis_reserved IN ITEMS --prefix --exec-prefix --libdir --includedir
                               --enable-shared --disable-shared
                               --enable-static --disable-static
                               CC CXX AR RANLIB)
    if(BLIS_CONFIGURE_OPTIONS MATCHES "(^|;)${_blis_reserved}")
        message(FATAL_ERROR
                "BLIS_CONFIGURE_OPTIONS must not contain '${_blis_reserved}': "
                "this module sets the install paths, static linkage and build "
                "tools itself.")
    endif()
endforeach()

# BLIS needs pthread whatever its threading model, and a static archive does
# not carry one.
find_package(Threads REQUIRED)

# Not CMAKE_MAKE_PROGRAM: BLIS builds with make under every generator.
find_program(BLIS_MAKE_EXECUTABLE NAMES gmake make REQUIRED
             DOC "The make executable used to build BLIS")

# Together these re-run this file in script mode, which the block at the top
# of the file handles.
set(_blis_make ${CMAKE_COMMAND} "-DMAKE_EXE=${BLIS_MAKE_EXECUTABLE}"
               "-DWORKDIR=<BINARY_DIR>")
set(_blis_script -P "${CMAKE_CURRENT_LIST_FILE}")

FetchContent_Declare(blis
    URL "${FAMLIES_TRACKED_BLIS_RELEASE_URL}"
    URL_HASH SHA256=${FAMLIES_TRACKED_BLIS_RELEASE_SHA256}
    DOWNLOAD_NO_PROGRESS TRUE)
FetchContent_MakeAvailable(blis)

file(READ "${blis_SOURCE_DIR}/build/version" BLIS_VERSION)
string(STRIP "${BLIS_VERSION}" BLIS_VERSION)

set(_blis_sha256 "${FAMLIES_TRACKED_BLIS_RELEASE_SHA256}")
if(FETCHCONTENT_SOURCE_DIR_BLIS)
    set(_blis_sha256 "")
endif()

# What the stamp records about the library being built. The hash identifies a
# downloaded archive; a tree supplied through FETCHCONTENT_SOURCE_DIR_BLIS has
# its hash blanked above, so only the path distinguishes it.
set(BLIS_BUILD_DIR "${FETCHCONTENT_BASE_DIR}/blis-build")
set(BLIS_STAMP_FILE "${FETCHCONTENT_BASE_DIR}/blis-ep/blis-provisioning.stamp")
set(_blis_stamp "version=${BLIS_VERSION}
sha256=${_blis_sha256}
source=${blis_SOURCE_DIR}
confname=${BLIS_CONFNAME}
options=${BLIS_CONFIGURE_OPTIONS}
")

# Re-running BLIS's configure is not enough to change what was configured. It
# mirrors the source tree into the build directory as symlinks and creates only
# the ones missing, so a build directory configured against an earlier source
# path keeps pointing there, and if that path is gone make finds no Makefile.
# Only an empty build directory fixes that, so a changed stamp wipes the build
# directory too.
set(_blis_stamp_old "")
if(EXISTS "${BLIS_STAMP_FILE}")
    file(READ "${BLIS_STAMP_FILE}" _blis_stamp_old)
endif()
if(NOT _blis_stamp_old STREQUAL _blis_stamp)
    file(REMOVE_RECURSE "${BLIS_BUILD_DIR}")
    file(WRITE "${BLIS_STAMP_FILE}" "${_blis_stamp}")
endif()

# Real paths: <INSTALL_DIR> is substituted in ExternalProject commands only.
set(BLIS_PREFIX "${FETCHCONTENT_BASE_DIR}/blis-install")
set(BLIS_INCLUDE_DIR "${BLIS_PREFIX}/include/blis")

set(BLIS_LIBRARY
    "${BLIS_PREFIX}/lib/\
${CMAKE_STATIC_LIBRARY_PREFIX}blis${CMAKE_STATIC_LIBRARY_SUFFIX}")

find_package_handle_standard_args(BLIS
    REQUIRED_VARS BLIS_LIBRARY
    VERSION_VAR   BLIS_VERSION)

if(BLIS_FOUND)
    ExternalProject_Add(blis_external
        # Already on disk; the empty commands keep a future GIT_* argument
        # from refetching it.
        SOURCE_DIR          "${blis_SOURCE_DIR}"
        DOWNLOAD_COMMAND    ""
        UPDATE_COMMAND      ""
        # Set explicitly, or the build tree would land under PREFIX beside
        # the logs.
        PREFIX              "${FETCHCONTENT_BASE_DIR}/blis-ep"
        BINARY_DIR          "${BLIS_BUILD_DIR}"
        INSTALL_DIR         "${BLIS_PREFIX}"
        # Runs in <BINARY_DIR>; confname has to come last.
        CONFIGURE_COMMAND   <SOURCE_DIR>/configure
                            --prefix=<INSTALL_DIR>
                            --disable-shared
                            ${BLIS_CONFIGURE_OPTIONS}
                            CC=${CMAKE_C_COMPILER}
                            CXX=${CMAKE_CXX_COMPILER}
                            AR=${CMAKE_AR}
                            RANLIB=${CMAKE_RANLIB}
                            ${BLIS_CONFNAME}
        BUILD_COMMAND       ${_blis_make} ${_blis_script}
        INSTALL_COMMAND     ${_blis_make} "-DTARGET=install" ${_blis_script}
        # Appears only once install runs; Ninja needs it declared to generate.
        INSTALL_BYPRODUCTS  "${BLIS_LIBRARY}"
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        LOG_OUTPUT_ON_FAILURE TRUE)

    # Every step, not just configure.
    foreach(step configure build install)
        ExternalProject_Add_StepDependencies(blis_external ${step}
                                             "${BLIS_STAMP_FILE}")
    endforeach()

    # An imported include directory must exist before generation.
    file(MAKE_DIRECTORY "${BLIS_INCLUDE_DIR}")

    add_library(BLIS::BLIS STATIC IMPORTED GLOBAL)
    set_target_properties(BLIS::BLIS PROPERTIES
        IMPORTED_LOCATION             "${BLIS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}")
    target_link_libraries(BLIS::BLIS INTERFACE Threads::Threads m)
    # Otherwise nothing triggers the build.
    add_dependencies(BLIS::BLIS blis_external)
endif()

mark_as_advanced(BLIS_MAKE_EXECUTABLE BLIS_INCLUDE_DIR BLIS_LIBRARY)
