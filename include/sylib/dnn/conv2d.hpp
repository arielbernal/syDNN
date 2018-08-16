#pragma once

#include <memory>
#include <sylib/tensor.hpp>
#include <sylib/size.hpp>

namespace sylib {
namespace dnn {

class Conv2DBase;

class Conv2D {
public:
  Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1});

  Conv2D(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1});
  ~Conv2D();
  void compile();
  void set_arguments();
  cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr);

  static Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1});
private:
  std::unique_ptr<Conv2DBase> _ptr;
};

} // dnn
} // sylib