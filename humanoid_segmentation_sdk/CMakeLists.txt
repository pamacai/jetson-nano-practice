cmake_minimum_required(VERSION 3.10)

project(humanoid_segmentation_sdk)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT paths
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu) # Update this path if necessary
set(TENSORRT_LIBRARY_DIR /usr/lib/aarch64-linux-gnu)     # Update this path if necessary

# Add TensorRT include and library paths
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
add_executable(demo_runner
    tools/demo_runner.cpp
    src/segmentation/TRTResNetSegmenter.cpp
)

# Link TensorRT and OpenCV libraries
target_link_libraries(demo_runner nvinfer nvinfer_plugin ${OpenCV_LIBS})

# Include directories
target_include_directories(demo_runner PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Packaging Configuration
include(CPack)

# Set package metadata
set(CPACK_PACKAGE_NAME "humanoid_segmentation_sdk")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Humanoid Segmentation SDK for image segmentation using TensorRT.")
set(CPACK_PACKAGE_VENDOR "YourCompany")
set(CPACK_PACKAGE_CONTACT "support@yourcompany.com")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "humanoid_segmentation_sdk")
