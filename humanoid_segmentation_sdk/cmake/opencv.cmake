include(ExternalProject)

set(OpenCV_VERSION 4.5.5)
set(OpenCV_REPO https://github.com/opencv/opencv.git)
set(OpenCV_CONTRIB_REPO https://github.com/opencv/opencv_contrib.git)

# Clone opencv_contrib repository first
ExternalProject_Add(
    OpenCVContrib
    PREFIX ${CMAKE_BINARY_DIR}/opencv_contrib
    GIT_REPOSITORY ${OpenCV_CONTRIB_REPO}
    GIT_TAG ${OpenCV_VERSION}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)


# Now build OpenCV, using the contrib modules path
ExternalProject_Add(
    OpenCV
    PREFIX ${CMAKE_BINARY_DIR}/opencv
    GIT_REPOSITORY ${OpenCV_REPO}
    GIT_TAG ${OpenCV_VERSION}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/opencv/install
        -DOPENCV_EXTRA_MODULES_PATH=${CMAKE_BINARY_DIR}/opencv_contrib/src/modules
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_DOCS=OFF
        -DWITH_CUDA=ON
        -DWITH_QT=OFF
        -DWITH_OPENGL=ON
        -DWITH_GSTREAMER=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_OPENEXR=OFF
        -DWITH_QT=OFF
        -DWITH_GTK=OFF
    DEPENDS OpenCVContrib
)

# Ensure OpenCVContrib is cloned before building OpenCV
add_dependencies(OpenCV OpenCVContrib)

# Add OpenCV include and library paths
set(OpenCV_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/opencv/install/include)
set(OpenCV_LIB_DIRS ${CMAKE_BINARY_DIR}/opencv/install/lib)
set(OpenCV_LIBS
    ${OpenCV_LIB_DIRS}/libopencv_core.so
    ${OpenCV_LIB_DIRS}/libopencv_imgproc.so
    ${OpenCV_LIB_DIRS}/libopencv_highgui.so
)

# Export OpenCV variables for use in the main CMakeLists.txt
set(OpenCV_FOUND TRUE)
set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBS})