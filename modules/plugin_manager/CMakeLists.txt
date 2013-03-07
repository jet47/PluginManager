set(target "opencv_plugin_manager")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories("${CMAKE_BINARY_DIR}/modules/core")

file(GLOB header_files "include/*.h" "include/*.hpp")
file(GLOB source_files "src/*.c" "src/*.cpp")
file(GLOB internal_header_files "src/*.h" "src/*.hpp")
set(all_build_files ${header_files} ${source_files} ${internal_header_files})

add_compiler_export_flags()

add_library(${target} ${all_build_files})

target_link_libraries(${target} ${POCO_LIBRARIES})

set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${bin_output_dir}
    LIBRARY_OUTPUT_DIRECTORY ${lib_output_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${lib_output_dir}
)

install(TARGETS ${target}
    RUNTIME DESTINATION ${bin_install_dir}
    LIBRARY DESTINATION ${lib_install_dir}
    ARCHIVE DESTINATION ${lib_install_dir}
)

install(FILES ${header_files} DESTINATION ${include_install_dir})
