set(target "cuda_arithm")

include_directories("${CMAKE_SOURCE_DIR}/modules/utility/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/plugin_manager/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/core/include")
include_directories("${CMAKE_BINARY_DIR}/modules/utility")

file(GLOB header_files "*.h" "*.hpp")
file(GLOB source_files "*.c" "*.cpp" "*.cu")
set(all_build_files ${header_files} ${source_files})

CUDA_ADD_LIBRARY(${target} SHARED ${all_build_files})

add_dependencies(${target} opencv_utility opencv_plugin_manager opencv_core)

target_link_libraries(${target} ${CUDA_LIBRARIES} opencv_utility opencv_plugin_manager opencv_core)

set_target_properties(${target} PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${bin_output_dir}/plugins/gpu/cuda"
    LIBRARY_OUTPUT_DIRECTORY "${lib_output_dir}/plugins/gpu/cuda"
    ARCHIVE_OUTPUT_DIRECTORY "${lib_output_dir}/plugins/gpu/cuda"
)
