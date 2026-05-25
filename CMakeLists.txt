cmake_minimum_required(VERSION 3.16)
project(bo C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CORE_SOURCES csrc/bo.c csrc/wl_api.c)

# Shared library
add_library(bo SHARED ${CORE_SOURCES})
target_include_directories(bo PUBLIC csrc)
target_compile_definitions(bo PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(bo PRIVATE -Wall -Wextra -Wpedantic -O2)
target_link_libraries(bo m)
set_target_properties(bo PROPERTIES OUTPUT_NAME bo)

# Test executable
if(BUILD_TESTING)
    add_executable(test_bo csrc/test_c.c)
    target_link_libraries(test_bo bo)
    target_include_directories(test_bo PRIVATE csrc)
    target_link_libraries(test_bo m)
endif()
