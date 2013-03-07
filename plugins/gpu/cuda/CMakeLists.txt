find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(main)
