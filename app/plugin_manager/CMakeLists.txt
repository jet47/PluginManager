set(target "plugin_manager")

include_directories("${CMAKE_SOURCE_DIR}/modules/utility/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/plugin_manager/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/core/include")

file(GLOB header_files "*.h" "*.hpp")
file(GLOB source_files "*.c" "*.cpp")
set(all_build_files ${header_files} ${source_files})

add_executable(${target} ${all_build_files})

add_dependencies(${target} opencv_utility opencv_plugin_manager opencv_core)

target_link_libraries(${target} opencv_utility opencv_plugin_manager opencv_core)

set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${bin_output_dir}
    LIBRARY_OUTPUT_DIRECTORY ${lib_output_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${lib_output_dir}
)
