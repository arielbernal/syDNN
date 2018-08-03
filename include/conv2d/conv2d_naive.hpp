#pragma once

#include <conv2d/conv2d_impl.hpp>

namespace syDNN {

class Conv2DNaive : public Conv2DImplBase {
public:
  Conv2DNaive(const cl::Context& context)
  : Conv2DImplBase(context)
  {
    add_kernel("opencl_kernels/conv2d_naive.cl", "convolution");
  }
private:
  static bool _registered;
};

bool Conv2DNaive::_registered = Conv2DFactory::register_implementation("Conv2DNaive", true,
                            [](const cl::Context& context) -> std::unique_ptr<Conv2DImplBase>
                              { return std::make_unique<Conv2DNaive>(context); }
                            );

} // namespace syDNN