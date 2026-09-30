# FindMArray
# ----------
#
# Provides MArray (https://github.com/MatthewsResearchGroup/marray) as the
# target MArray::marray. MArray ships a CMake package config, which this module
# looks for first. Failing that, it downloads and builds MArray as a subproject:
# the release that external/versions.cmake pins as
# FAMLIES_TRACKED_MARRAY_RELEASE_*, which must be included first.

if(NOT TARGET MArray::marray)
    include(FetchContent)

    # Not in the install prefix: a fetched MArray installs itself there, and a
    # later configure would take that copy over the pinned one.
    FetchContent_Declare(MArray
        URL "${FAMLIES_TRACKED_MARRAY_RELEASE_URL}"
        URL_HASH SHA256=${FAMLIES_TRACKED_MARRAY_RELEASE_SHA256}
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        FIND_PACKAGE_ARGS CONFIG NO_CMAKE_INSTALL_PREFIX)

    # MArray's install directories default to absolute paths under the
    # configure-time CMAKE_INSTALL_PREFIX, which `cmake --install --prefix`
    # can't move, so famlies's package would name an MArray somewhere else.
    # Relative ones install it beside famlies.
    block()
        include(GNUInstallDirs)
        set(INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
        set(INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
        FetchContent_MakeAvailable(MArray)
    endblock()
endif()

# The target, not MArray_FOUND, which only the config search sets: an MArray
# built as a subproject leaves it unset, or false if that search ran first.
set(MArray_TARGET "")
if(TARGET MArray::marray)
    set(MArray_TARGET MArray::marray)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MArray REQUIRED_VARS MArray_TARGET)
