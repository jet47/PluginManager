cmake_minimum_required(VERSION 2.8.10)

project(opencv_test)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_PLUGINS "Build plugins" OFF)

set(bin_output_dir "${CMAKE_BINARY_DIR}/bin")
set(lib_output_dir "${CMAKE_BINARY_DIR}/lib")

include(GenerateExportHeader)

find_package(POCO REQUIRED COMPONENTS XML Util)
include_directories(${POCO_INCLUDE_DIRS})

add_subdirectory(modules)
add_subdirectory(app)
add_subdirectory(sample)

if(BUILD_PLUGINS)
    add_subdirectory(plugins)
endif()
