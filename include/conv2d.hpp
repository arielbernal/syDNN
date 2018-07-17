#pragma once

#include <tensor.hpp>

namespace syDNN {

// struct Kernel {
//   cl::Kernel kernel;
//   cl::Range global;
//   cl::Range local;
//   cl::Range offset;
// }


// Kernel get_kernel(cl::Context context, const std::string& sourceCode, const std::string& kernelName, const std::string& compileOptions) {
//   Kernel k;
//   cl_int err = CL_SUCCESS;
//   Program program = Program(context, sourceCode, false);
//   err = program.build(compileOptions.c_str());
//   k.kernel(Kernel(program, kernelName));
//   return k;
// }

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