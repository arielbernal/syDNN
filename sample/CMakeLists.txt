cmake_minimum_required(VERSION 2.8)

set(TARGET_NAME sptests)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCV)
find_package(OpenCL REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories( BEFORE
                     ${CMAKE_CURRENT_SOURCE_DIR}
                     ${PROJECT_SOURCE_DIR}/include
                     ${OpenCL_INCLUDE_DIR} )

add_executable ( ${TARGET_NAME} main.cpp )

target_link_libraries( ${TARGET_NAME} ${OpenCL_LIBRARY} ${OpenCV_LIBS} -lpthread )