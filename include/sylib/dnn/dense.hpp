#pragma once

#include <memory>

#include <sylib/syop.hpp>
#include <sylib/tensor.hpp>
#include <sylib/size.hpp>

namespace sylib {
namespace dnn {

class Dense : public Operation {
public:
  Dense(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
          const Tensor& weights, const Tensor& bias);
  static Size output_shape(const Tensor& input, const Tensor& weights);
  static std::vector<Type> input_type(const std::string& name);
};

} // dnn
} // sylib