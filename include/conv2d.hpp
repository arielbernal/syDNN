#pragma once
#include <fstream>
#include <iostream>
#include <unordered_map>

#include <conv2d/conv2d_naive.hpp>
#include <conv2d/conv2d_3x3.hpp>
#include <conv2d/conv2d_bfyx_os_iyx_osv16.hpp>

namespace syDNN {

class Conv2D {
public:
  Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1})
  : _ptr(Conv2DFactory::create(name, context, input, output, weights, bias, padding, stride, dilation))
  {}

  Conv2D(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1})
  : _ptr(Conv2DFactory::create("", context, input, output, weights, bias, padding, stride, dilation))
  {}

  void compile() { _ptr->compile(); }
  void set_arguments() { _ptr->set_arguments(); }
  cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    return _ptr->enqueue(queue, events, event);
  }

  static Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    int input_padding_w = 0;
    int input_padding_h = 0;
    if (padding == sy_same) {
      input_padding_h = weights.shape(2) / 2;
      input_padding_w = weights.shape(3) / 2;
    }

    int32_t out_h = 1 + (int(input.shape(2)) + 2 * input_padding_h - ((weights.shape(2) - 1) * dilation[0] + 1)) / stride[0];
    int32_t out_w = 1 + (int(input.shape(3)) + 2 * input_padding_w - ((weights.shape(3) - 1) * dilation[1] + 1)) / stride[1];
    Size ret {input.shape(0), weights.shape(0), out_h, out_w};
    return ret;
  }
private:
  Conv2DPtr _ptr;
};

}