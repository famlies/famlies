find_package(Catch2 REQUIRED Catch2::Catch2WithMain)
include(Catch)

set(FAMLIES_TEST_WARNINGS
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wextra>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wpedantic>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wshadow>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wconversion>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wsign-conversion>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wold-style-cast>)

add_executable(famlies_tests
    blas/gemm.cpp
    blas/herk.cpp
    blas/trsm.cpp
    core/conj.cpp
    core/control.cpp
    core/operand.cpp
    core/precondition.cpp
    core/precondition_off.cpp
    core/sweep.cpp
    dec/chol.cpp
    leaf/blis.cpp
    leaf/custom.cpp
    leaf/reference.cpp
    support/support.cpp)
target_include_directories(famlies_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(famlies_tests PRIVATE famlies Catch2::Catch2WithMain)
target_compile_options(famlies_tests PRIVATE ${FAMLIES_TEST_WARNINGS})

catch_discover_tests(famlies_tests)

# One ctest per case of precondition_abort.cpp, passing only on famlies's
# message for the case's op, since BLIS's own failed checks abort too.
# Without checks there is nothing to abort.
if(NOT FAMLIES_NO_PRECONDITION_CHECKS)
  add_executable(precondition_abort precondition_abort.cpp)
  target_link_libraries(precondition_abort PRIVATE famlies)
  target_compile_options(precondition_abort PRIVATE ${FAMLIES_TEST_WARNINGS})

  # The cases are the quoted "<op>.<case>" names in its table.
  set(path "${CMAKE_CURRENT_SOURCE_DIR}/precondition_abort.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
  file(READ "${path}" text)
  string(REGEX MATCHALL "\"[a-z_]+\\.[a-z_]+\"" cases "${text}")
  string(REPLACE "\"" "" cases "${cases}")

  foreach(case IN LISTS cases)
    string(REGEX REPLACE "\\..*$" "" op "${case}")
    add_test(NAME "precondition_abort.${case}"
                 COMMAND precondition_abort "${case}")
    set_tests_properties("precondition_abort.${case}" PROPERTIES
            PASS_REGULAR_EXPRESSION "famlies::${op}: precondition failed")
  endforeach()
endif()
