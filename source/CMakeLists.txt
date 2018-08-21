cmake_minimum_required(VERSION 3.9)

set(TARGET_NAME sylib)

find_package(OpenCL REQUIRED)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# =================================== Link targets and dependencies ====================================
# Main shared library.
file(GLOB SOURCES "*.cpp" "dnn/*.cpp") # yes I'm lazy
add_library("${TARGET_NAME}" SHARED ${SOURCES})

set_target_properties("${TARGET_NAME}" PROPERTIES VERSION "${PROJECT_VERSION}")
set_target_properties("${TARGET_NAME}" PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_include_directories("${TARGET_NAME}" PUBLIC  "${PROJECT_SOURCE_DIR}/include")
target_include_directories("${TARGET_NAME}" PRIVATE "${PROJECT_SOURCE_DIR}/source/include")
target_include_directories("${TARGET_NAME}" PRIVATE ${OpenCL_INCLUDE_DIR})

target_link_libraries("${TARGET_NAME}" ${OpenCL_LIBRARY} -lpthread)

# ========================================== Installation ==============================================
# API headers.
install(DIRECTORY ${CMAKE_BINARY_DIR}
    DESTINATION "include/sylib"
    FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
  )
# Binaries.
install(TARGETS "${TARGET_NAME}"
    ARCHIVE DESTINATION "lib"
    LIBRARY DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "include/sylib"
  )