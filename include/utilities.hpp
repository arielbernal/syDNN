#pragma once
#include <type_traits>
#include <fstream>
#include <iostream>
#include <CL/cl.hpp>

namespace syDNN
{

inline size_t align(size_t val, size_t alignment)
{
    size_t diff = val % alignment;
    return (diff == 0) ? val : val + alignment - diff;
}

inline size_t pad(size_t val, size_t alignment)
{
    size_t diff = val % alignment;
    return (diff == 0) ? 0 : alignment - diff;
}

inline bool is_aligned(size_t val, size_t alignment)
{
  return !(val % alignment);
}

template <bool...> struct bool_pack;
template <bool... bs>
using all_true = std::is_same<bool_pack<true, bs...>, bool_pack<bs...,true>>;

template <typename T, typename... Ts>
using all_same = all_true<std::is_same<T, Ts>::value...>;

template <typename... Ts>
using all_integral = all_true<std::is_integral<Ts>::value...>;

template <bool B, typename T = void>
struct disable_if { typedef T type; };
template <typename T>
struct disable_if<true, T> {};


inline const char* OpenCLErrorString(cl_int err)
{
    switch(err)
    {
      case CL_SUCCESS:                            return "CL_SUCCESS";
      case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
      case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
      case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
      case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
      case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
      case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
      case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
      case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
      case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
      case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
      case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
      case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
      case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
      case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
      case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
      case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
      case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
      case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
      case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
      case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
      case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
      case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
      case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
      case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
      case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
      case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
      case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
      case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
      case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
      case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
      case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
      case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
      case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
      case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
      case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
      case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
      case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
      case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
      default: break;
    }
    return "CL_UNKOWN_ERROR";
}

inline std::string load_program(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open())
    throw std::ios_base::failure("load_program -> File not found: " + filename);

  std::string ret;
  ret.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  return ret;
}

inline cl::Kernel compile_kernel(cl::Context context, const std::string& sourceCode, const std::string& kernelName,
                                  const std::string& compileOptions = "") {
  cl_int err = CL_SUCCESS;
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Program program = cl::Program(context, sourceCode, false);
  err = program.build(compileOptions.c_str());
  if (err != CL_SUCCESS) {
    for (auto& e : devices) {
      std::string device_name = e.getInfo<CL_DEVICE_NAME>();
      std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(e);
      std::cout << "Error compiling kernel : " << kernelName << " (Device : " << device_name << ")" << std::endl;
      std::cout << log << std::endl;
    }
    throw std::runtime_error("compile_kernel -> build error");
  }
  return cl::Kernel(program, kernelName.c_str());
}


namespace detail
{

// TODO: fix error handling with optimal exceptions
static inline cl_int errorHandler(cl_int err, const char* errorStr = nullptr)
{
#if defined(__CL_ENABLE_EXCEPTIONS)
    if(err != CL_SUCCESS)
        throw Error(err, errStr);
    return err;
#else
    errorStr = nullptr;
    return err;
#endif // __CL_ENABLE_EXCEPTIONS
}

} // namespace detail



} // namespace syDNN