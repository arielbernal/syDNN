#pragma once

#include <memory>

#include <sylib/syop.hpp>
#include <sylib/tensor.hpp>
#include <sylib/size.hpp>

namespace sylib {
namespace dnn {


class Conv2DBase;

class Conv2D : Operation {
public:
  Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
          const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
          const Size& stride = {1, 1}, const Size& dilation = {1, 1});
  ~Conv2D();
  static Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1});
  void compile() override;
  void set_arguments() override;
  cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) override;
private:
  std::unique_ptr<Conv2DBase> _impl;
};

} // dnn
} // sylib