find_path( CL_INCLUDE_DIR CL/cl.h
    HINTS /opt/intel/opencl/include )
find_library( CL_LIB_NAME NAMES OpenCL
    HINTS /opt/intel/opencl/)

add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=200)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CL_INCLUDE_DIR}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


file(GLOB TEST_SRC "*.cpp" "./common/*.cpp")

add_executable(tests ${TEST_SRC})
target_link_libraries(tests gtest gmock_main ${CL_LIB_NAME})
add_test(NAME tests COMMAND test)