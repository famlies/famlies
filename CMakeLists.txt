cmake_minimum_required(VERSION 3.31)

project(famlies
    VERSION 0.1.0
    DESCRIPTION
    "Framework for Advanced (Multi)Linear\
     Infrastructure in Engineering and Science"
    LANGUAGES C CXX)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include("${PROJECT_SOURCE_DIR}/external/versions.cmake")

find_package(BLIS REQUIRED)
find_package(MArray REQUIRED)

add_library(famlies INTERFACE)
add_library(famlies::famlies ALIAS famlies)
target_compile_features(famlies INTERFACE cxx_std_23)
target_link_libraries(famlies INTERFACE BLIS::BLIS MArray::marray)
target_sources(famlies INTERFACE
    FILE_SET HEADERS
    FILES
        famlies/famlies.hpp
        famlies/blas/gemm.hpp
        famlies/blas/herk.hpp
        famlies/blas/trsm.hpp
        famlies/core/conj.hpp
        famlies/core/control.hpp
        famlies/core/error.hpp
        famlies/core/operand.hpp
        famlies/core/precondition.hpp
        famlies/core/sweep.hpp
        famlies/core/tags.hpp
        famlies/dec/chol.hpp
        famlies/leaf/blis.hpp
        famlies/leaf/reference.hpp)

option(FAMLIES_NO_PRECONDITION_CHECKS
       "Compile out famlies's precondition checks" OFF)
if(FAMLIES_NO_PRECONDITION_CHECKS)
    target_compile_definitions(famlies INTERFACE
        FAMLIES_NO_PRECONDITION_CHECKS)
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()
if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

option(FAMLIES_BUILD_BENCHMARKS
       "Build the benchmark against libflame" OFF)
if(FAMLIES_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

option(FAMLIES_INSTALL "Generate famlies's install and export rules"
       ${PROJECT_IS_TOP_LEVEL})

if(FAMLIES_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(FAMLIES_INSTALL_BLIS_LIBDIR "${CMAKE_INSTALL_LIBDIR}/famlies")
    set(FAMLIES_INSTALL_BLIS_INCLUDEDIR
        "${CMAKE_INSTALL_INCLUDEDIR}/famlies/blis")

    # Unset when BLIS::BLIS was defined by a superproject
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
