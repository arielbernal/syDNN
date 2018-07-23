#pragma once
#include <fstream>
#include <iostream>
#include <tensor.hpp>

namespace syDNN {

struct SyKernel {
  cl::Kernel kernel;
  cl::NDRange gws;
  cl::NDRange lws;
  cl::NDRange offset;
};

inline std::string load_kernel(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open())
    throw std::ios_base::failure("load_kernel -> File not found: " + filename);

  std::string ret;
  ret.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  return ret;
}

inline SyKernel get_kernel(cl::Context context, const std::string& sourceCode, const std::string& kernelName,
                  const std::string& compileOptions = "") {
  SyKernel k;
  cl_int err = CL_SUCCESS;
  cl::Program program = cl::Program(context, sourceCode, false);
  err = program.build(compileOptions.c_str());
  k.kernel = cl::Kernel(program, kernelName.c_str());
  return k;
}

// Kernel conv_2d(cl::Context context, const Tensor& X, const Tensor& Weights, const Tensor& bias, const Size& stride, Tensor& Y)
// {
//   // based on Input.shape/padding and W.shape/padding select the best kernel implementation
//   // kernelSelector()
//   Kernel k = get_kernel(context, sourceCode, kernelName, compileOptions);
//   k.kernel.setArg(0, X.buffer());
//   k.kernel.setArg(1, Weights.buffer());
//   k.kernel.setArg(2, bias.buffer());
//   k.kernel.setArg(3, Y.buffer());
//   return k;
// }


} // namespace syDNN