cmake_minimum_required(VERSION 3.10)
project(stl_with_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB CUDA_SOURCES "*.cu")
foreach(CUDA_FILE ${CUDA_SOURCES})
    get_filename_component(EXECUTABLE_NAME ${CUDA_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${CUDA_FILE})
endforeach()

set_target_properties(stl_with_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/../../build")
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    COMMENT "Cleaning all build files"
)