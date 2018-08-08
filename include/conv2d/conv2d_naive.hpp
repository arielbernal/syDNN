#pragma once

#include <conv2d/conv2d_impl.hpp>

namespace syDNN {

class Conv2DNaive : public Conv2DBase {
public:
  Conv2DNaive(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding, const Size& stride, const Size& dilation)
  : Conv2DBase(context, input, output, weights, bias, padding, stride, dilation)
  {
    add_kernel("opencl_kernels/conv2d_naive.cl", "convolution");
  }

  Layout input_layout() override { return sy_nchw; }
  Layout weights_layout() override { return sy_nchw; }
  Layout output_layout() override { return sy_nchw; }

  std::vector<Type> input_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> output_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> weights_type() override { return {sy_fp32, sy_fp16}; }

  Size output_shape() override {
    int input_padding_w = 0;
    int input_padding_h = 0;
    if (_padding == sy_same) {
      input_padding_h = _weights->shape(2) / 2;
      input_padding_w = _weights->shape(3) / 2;
    }

    int32_t out_h = 1 + (int(_input->shape(2)) + 2 * input_padding_h - ((_weights->shape(2) - 1) * _dilation[0] + 1)) / _stride[0];
    int32_t out_w = 1 + (int(_input->shape(3)) + 2 * input_padding_w - ((_weights->shape(3) - 1) * _dilation[1] + 1)) / _stride[1];
    Size ret {_input->shape(0), _weights->shape(0), out_h, out_w};
    return ret;
  }

  void compile() override {
    std::stringstream preamble;
    preamble << kernel_define("FUNC(x)", "x");
    preamble << kernel_define("FUNC_CALL(x)", "x");
    preamble << kernel_define("KERNEL(x)", "void __kernel x");
    preamble << getTensor4DOption("INPUT", *_input);
    preamble << getTensor4DOption("FILTER", *_weights);
    preamble << getTensor4DOption("OUTPUT", *_output);
    preamble << kernel_define("COMPUTE_TYPE", "float");

    preamble << kernel_define("COMPUTE_TYPE", "float");
    preamble << kernel_define("STRIDE_Y", _stride[0]);
    preamble << kernel_define("STRIDE_X", _stride[1]);
    preamble << kernel_define("DILATION_Y", _dilation[0]);
    preamble << kernel_define("DILATION_X", _dilation[1]);
    preamble << kernel_define("INPUT_Y_OFFSET", 0);
    preamble << kernel_define("INPUT_X_OFFSET", 0);
    if (_bias) {
      preamble << kernel_define("BIAS_TYPE", "float");
      preamble << kernel_define("BIAS_TERM", "true");
    }
    Kernel& k = kernel();
    k.compile(preamble.str(), "-I opencl_kernels");
  }

  void set_arguments() override {
    Kernel& k = kernel();
    k.global_work_size(cl::NDRange(_output->shape(3), _output->shape(2), _output->shape(1) * _output->shape(0)));
    k.add_argument(_input->buffer());
    k.add_argument(_output->buffer());
    k.add_argument(_weights->buffer());
    if (_bias)
      k.add_argument(_bias->buffer());
  }

private:
  static bool _registered;
};

bool Conv2DNaive::_registered = Conv2DFactory::register_implementation<Conv2DNaive>("Conv2DNaive");

} // namespace syDNN