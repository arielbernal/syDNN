cmake_minimum_required(VERSION 3.9)
set(TARGET_NAME tests)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
file(GLOB TEST_SRC "*.cpp" "./common/*.cpp")

add_executable(${TARGET_NAME} ${TEST_SRC})
add_dependencies("${TARGET_NAME}" sylib)

target_link_libraries(${TARGET_NAME} gtest gmock_main ${OpenCL_LIBRARY}  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libsylib.so")
target_include_directories("${TARGET_NAME}" PRIVATE  "${PROJECT_SOURCE_DIR}/include")
target_include_directories("${TARGET_NAME}" PRIVATE  "${CMAKE_BINARY_DIR}/googletest-src/googletest/include")
target_include_directories("${TARGET_NAME}" PRIVATE  "${PROJECT_SOURCE_DIR}/tests")
target_include_directories("${TARGET_NAME}" PRIVATE  "${OpenCL_INCLUDE_DIR}")

add_test(NAME ${TARGET_NAME} COMMAND test)