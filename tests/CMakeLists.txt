cmake_minimum_required(VERSION 2.8)
set(TARGET_NAME tests)


set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120 -DCL_HPP_TARGET_OPENCL_VERSION=120)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCL_INCLUDE_DIR}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


file(GLOB TEST_SRC "*.cpp" "./common/*.cpp")

add_executable(${TARGET_NAME} ${TEST_SRC})
target_link_libraries(${TARGET_NAME} gtest gmock_main ${OpenCL_LIBRARY})

add_test(NAME ${TARGET_NAME} COMMAND test)