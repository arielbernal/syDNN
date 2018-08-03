#pragma once

#include <implementation.hpp>

namespace syDNN {

class Conv2DImplBase : public ImplementationBase {
public:
  Conv2DImplBase(const cl::Context& context)
  : ImplementationBase(context)
  {}

  virtual Layout input_layout() { return sy_nchw; }
  virtual Layout weights_layout() { return sy_nchw; }
  virtual Layout output_layout() { return sy_nchw; }

  virtual std::vector<Type> input_type() { return {sy_fp32, sy_fp16}; }
  virtual std::vector<Type> output_type() { return {sy_fp32, sy_fp16}; }
  virtual std::vector<Type> weights_type() { return {sy_fp32, sy_fp16}; }

  virtual Size output_shape() {
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

  virtual void bind(const Tensor& input, const Tensor& output, const Tensor& weights, const Tensor& bias = Tensor(),
                     const Padding& padding = sy_valid, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    _input = &input;
    _output = &output;
    _weights = &weights;
    _bias = (bias.dim() > 0 ? &bias : nullptr);
    _padding = padding;
    _stride = stride;
    _dilation = dilation;
    // validate here shapes/layouts/types here
  }

  virtual void bind(const Tensor& input, const Tensor& output, const Tensor& weights,
                     const Padding& padding = sy_valid, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    bind(input, output, weights, Tensor(), padding, stride, dilation);
  }

  virtual void compile() {
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

  virtual void set_arguments() {
    Kernel& k = kernel();
    k.global_work_size(cl::NDRange(_output->shape(3), _output->shape(2), _output->shape(1) * _output->shape(0)));
    k.add_argument(_input->buffer());
    k.add_argument(_output->buffer());
    k.add_argument(_weights->buffer());
    if (_bias)
      k.add_argument(_bias->buffer());
  }
  virtual cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    Kernel& k = kernel();
    return k.enqueue(queue, events, event);
  }
protected:
  const Tensor* _input;
  const Tensor* _output;
  const Tensor* _weights;
  const Tensor* _bias;
  Padding _padding;
  Size _stride;
  Size _dilation;
  Size _input_offset;
  Size _output_offset;
};

class Conv2DFactory : public ImplementationFactory<Conv2DImplBase> {};
using Conv2D = std::unique_ptr<Conv2DImplBase>;

}