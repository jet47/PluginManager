set(target "cuda_npp")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/modules/plugin_manager/include")
include_directories("${CMAKE_SOURCE_DIR}/modules/core/include")

include_directories("${CMAKE_BINARY_DIR}/modules/core")

file(GLOB header_files "*.h" "*.hpp")
file(GLOB source_files "*.c" "*.cpp")
set(all_build_files ${header_files} ${source_files})

add_compiler_export_flags()

add_library(${target} SHARED ${all_build_files})

generate_export_header(${target})

add_dependencies(${target} opencv_plugin_manager opencv_core)

target_link_libraries(${target} ${POCO_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY} opencv_plugin_manager opencv_core)

set_target_properties(${target} PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${bin_output_dir}/plugins/gpu/cuda"
    LIBRARY_OUTPUT_DIRECTORY "${lib_output_dir}/plugins/gpu/cuda"
    ARCHIVE_OUTPUT_DIRECTORY "${lib_output_dir}/plugins/gpu/cuda"
)

install(TARGETS ${target}
    RUNTIME DESTINATION "${bin_install_dir}/plugins/gpu/cuda"
    LIBRARY DESTINATION "${lib_install_dir}/plugins/gpu/cuda"
    ARCHIVE DESTINATION "${lib_install_dir}/plugins/gpu/cuda"
)
