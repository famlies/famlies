cmake_minimum_required(VERSION 3.31)

project(famlies
    VERSION 0.1.0
    DESCRIPTION
    "Framework for Advanced (Multi)Linear\
     Infrastructure in Engineering and Science"
    LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

find_package(BLIS REQUIRED)

add_library(famlies INTERFACE)
add_library(famlies::famlies ALIAS famlies)
target_compile_features(famlies INTERFACE cxx_std_20)
target_link_libraries(famlies INTERFACE BLIS::BLIS)
target_sources(famlies INTERFACE
    FILE_SET HEADERS
    BASE_DIRS "${PROJECT_SOURCE_DIR}"
    FILES "${PROJECT_SOURCE_DIR}/famlies/famlies.hpp")

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

option(FAMLIES_INSTALL "Generate famlies's install and export rules"
       ${PROJECT_IS_TOP_LEVEL})

if(FAMLIES_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(FAMLIES_INSTALL_BLIS_LIBDIR "${CMAKE_INSTALL_LIBDIR}/famlies")
    set(FAMLIES_INSTALL_BLIS_INCLUDEDIR
        "${CMAKE_INSTALL_INCLUDEDIR}/famlies/blis")

    # Unset when BLIS::BLIS was defined before find_package(BLIS) reached its
    # own body -- a superproject's, not ours, and not ours to install.
    if(NOT BLIS_LIBRARY OR NOT BLIS_INCLUDE_DIR)
        message(FATAL_ERROR
                "FAMLIES_INSTALL needs the BLIS this project builds, but "
                "BLIS::BLIS came from elsewhere. Install that BLIS as its own "
                "package and turn FAMLIES_INSTALL off.")
    endif()
    install(FILES "${BLIS_LIBRARY}"
            DESTINATION "${FAMLIES_INSTALL_BLIS_LIBDIR}")
    install(DIRECTORY "${BLIS_INCLUDE_DIR}/"
            DESTINATION "${FAMLIES_INSTALL_BLIS_INCLUDEDIR}")

    install(TARGETS famlies
            EXPORT famlies-targets
            FILE_SET HEADERS)
    install(EXPORT famlies-targets
            FILE famlies-targets.cmake
            NAMESPACE famlies::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/famlies")

    cmake_path(GET BLIS_LIBRARY FILENAME FAMLIES_BLIS_LIBRARY_NAME)
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/famlies-config.cmake.in"
        "${PROJECT_BINARY_DIR}/famlies-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/famlies"
        PATH_VARS FAMLIES_INSTALL_BLIS_LIBDIR FAMLIES_INSTALL_BLIS_INCLUDEDIR)

    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/famlies-config-version.cmake"
        COMPATIBILITY SameMinorVersion)

    install(FILES "${PROJECT_BINARY_DIR}/famlies-config.cmake"
                  "${PROJECT_BINARY_DIR}/famlies-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/famlies")
endif()
