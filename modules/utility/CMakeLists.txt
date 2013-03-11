set(target "opencv_utility")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads REQUIRED)

file(GLOB header_files "include/*.h" "include/*.hpp")
file(GLOB source_files "src/*.c" "src/*.cpp")
file(GLOB internal_header_files "src/*.h" "src/*.hpp")
set(all_build_files ${header_files} ${source_files} ${internal_header_files})

add_library(${target} SHARED ${all_build_files})

target_link_libraries(${target} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${bin_output_dir}
    LIBRARY_OUTPUT_DIRECTORY ${lib_output_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${lib_output_dir}
)
