cmake_minimum_required(VERSION 2.8.10)

project(plugin_manager_test)

option(BUILD_PLUGINS "Build plugins" OFF)

set(bin_output_dir "${CMAKE_BINARY_DIR}/bin")
set(lib_output_dir "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(modules)
add_subdirectory(app)
add_subdirectory(sample)

if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()
