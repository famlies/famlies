add_executable(blis_smoke blis_smoke.cpp)
target_link_libraries(blis_smoke PRIVATE famlies)

add_test(NAME blis_smoke COMMAND blis_smoke)
