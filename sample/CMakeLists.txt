cmake_minimum_required(VERSION 3.9)

set(TARGET_NAME sample)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCV)
find_package(OpenCL REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

file(GLOB SOURCES "*.cpp")
add_executable("${TARGET_NAME}" ${SOURCES})

target_include_directories("${TARGET_NAME}" PUBLIC  "${PROJECT_SOURCE_DIR}/include")
target_include_directories("${TARGET_NAME}" PUBLIC  "${CMAKE_BINARY_DIR}/googletest-src/googletest/include")
target_include_directories("${TARGET_NAME}" PUBLIC  "${PROJECT_SOURCE_DIR}/tests/common")
target_include_directories("${TARGET_NAME}" PUBLIC  "${OpenCL_INCLUDE_DIR}")

target_link_libraries("${TARGET_NAME}" ${OpenCL_LIBRARY} ${OpenCV_LIBS} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libsylib.so")


