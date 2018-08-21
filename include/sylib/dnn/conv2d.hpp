#pragma once

#include <memory>

#include <sylib/syop.hpp>
#include <sylib/tensor.hpp>
#include <sylib/size.hpp>

namespace sylib {
namespace dnn {

class Conv2D : public Operation {
public:
  Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
          const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
          const Size& stride = {1, 1}, const Size& dilation = {1, 1});
  static Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1});
};

} // dnn
} // sylib