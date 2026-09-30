# FindCatch2
# ----------
#
# Provides Catch2 (https://github.com/catchorg/Catch2).
# Variables FAMLIES_TRACKED_CATCH2_RELEASE_* must have been defined.
string(REGEX MATCH "^[0-9]+"
       catch2_major "${FAMLIES_TRACKED_CATCH2_RELEASE_TAG}")

include(FetchContent)

FetchContent_Declare(Catch2
    URL "${FAMLIES_TRACKED_CATCH2_RELEASE_URL}"
    URL_HASH SHA256=${FAMLIES_TRACKED_CATCH2_RELEASE_SHA256}
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL
    SYSTEM
    FIND_PACKAGE_ARGS "${catch2_major}" CONFIG)

FetchContent_MakeAvailable(Catch2)

set(Catch2_TARGET "")
if(TARGET Catch2::Catch2)
  set(Catch2_TARGET Catch2::Catch2)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Catch2 REQUIRED_VARS Catch2_TARGET)
