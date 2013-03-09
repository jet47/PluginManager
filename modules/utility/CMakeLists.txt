set(target "opencv_utility")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

file(GLOB header_files "include/*.h" "include/*.hpp")
file(GLOB source_files "src/*.c" "src/*.cpp")
file(GLOB internal_header_files "src/*.h" "src/*.hpp")
set(all_build_files ${header_files} ${source_files} ${internal_header_files})

add_compiler_export_flags()

add_library(${target} SHARED ${all_build_files})

generate_export_header(${target} BASE_NAME opencv)

target_link_libraries(${target} ${POCO_LIBRARIES})

set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${bin_output_dir}
    LIBRARY_OUTPUT_DIRECTORY ${lib_output_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${lib_output_dir}
)
