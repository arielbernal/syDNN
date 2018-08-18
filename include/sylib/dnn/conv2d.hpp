#pragma once

#include <memory>

#include <sylib/tensor.hpp>
#include <sylib/size.hpp>
#include <sylib/syop.hpp>

namespace sylib {
namespace dnn {

class Conv2D {
public:
  Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
          const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
          const Size& stride = {1, 1}, const Size& dilation = {1, 1});
  ~Conv2D();
  static Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1});
private:
  class Conv2DImpl;
  std::unique_ptr<Conv2DImpl> _impl;
};

} // dnn
} // sylib