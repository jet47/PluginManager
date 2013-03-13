set(target "cuda")

include_directories("${CMAKE_SOURCE_DIR}/modules/utility/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/plugin_manager/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/core/include")

file(GLOB header_files "*.h" "*.hpp")
file(GLOB source_files "*.c" "*.cpp")
set(all_build_files ${header_files} ${source_files})

add_library(${target} SHARED ${all_build_files})

add_dependencies(${target} opencv_utility opencv_plugin_manager opencv_core)

target_link_libraries(${target} opencv_utility opencv_plugin_manager opencv_core)

set_target_properties(${target} PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${bin_output_dir}/plugins"
    LIBRARY_OUTPUT_DIRECTORY "${lib_output_dir}/plugins"
    ARCHIVE_OUTPUT_DIRECTORY "${lib_output_dir}/plugins"
)
