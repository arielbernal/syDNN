cmake_minimum_required(VERSION 2.8)

set(TARGET_NAME sptests)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package( OpenCV )

find_path( CL_INCLUDE_DIR CL/cl.h
    HINTS /opt/intel/opencl/include )
find_library( CL_LIB_NAME NAMES OpenCL
    HINTS /opt/intel/opencl/)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=200)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories( BEFORE
                     ${CMAKE_CURRENT_SOURCE_DIR}
                     ${PROJECT_SOURCE_DIR}/include
                     ${CL_INCLUDE_DIR} )

add_executable ( ${TARGET_NAME} main.cpp )

target_link_libraries( ${TARGET_NAME} ${CL_LIB_NAME} ${OpenCV_LIBS} -lpthread )

# Workaround: This is needed to avoid the error: undefined symbol: __intel_avx_rep_memcpy
# I don't think it should be required.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set( AVX_REP_MEMCPY_WORKAROUND CACHE BOOL "Workaround for \"undefined symbol: __intel_avx_rep_memcpy\"" )
    find_library( IRC_LIB NAMES irc
        HINTS /opt/intel/compilers_and_libraries/linux/lib/intel64_lin )
    if( AVX_REP_MEMCPY_WORKAROUND )
        message( STATUS "Enabling workaround for \"undefined symbol: __intel_avx_rep_memcpy\"" )
        message( STATUS "irc Lib is: ${IRC_LIB}" )
        target_link_libraries( ${TARGET_NAME} ${IRC_LIB} )
    endif()
endif()
#set(ENABLE_ITT true)
# ITT Support (optional)
# Linux Only... for now.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set( ENABLE_ITT CACHE BOOL "Enable ITT (Instrumentation Tracing Technology) API Support" )
    find_path( VTUNE_INCLUDE_DIR ittnotify.h
        HINTS /opt/intel/vtune_amplifier_xe/include )
    find_library( VTUNE_ITTNOTIFY_LIB NAMES ittnotify
        HINTS /opt/intel/vtune_amplifier_xe/lib64 )
    find_package( Threads )
    if( ENABLE_ITT )
        message("ITT Enabbled-------------------------")
        add_definitions("-DUSE_ITT")
        include_directories( ${VTUNE_INCLUDE_DIR} )
        message( STATUS "VTune ITTNotify Lib is: ${VTUNE_ITTNOTIFY_LIB}" )
        target_link_libraries( ${TARGET_NAME} ${VTUNE_ITTNOTIFY_LIB} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}  )
    endif()
endif()
