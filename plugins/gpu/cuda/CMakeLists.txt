find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(main)
add_subdirectory(npp)
add_subdirectory(arithm)
