#pragma once

#include <implementation.hpp>

namespace syDNN {

class Conv2DImplBase : public ImplementationBase {
public:
  Conv2DImplBase(const std::string& programFilename)
  : ImplementationBase(programFilename)
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

  virtual void bind(cl::Context context, const Tensor& input, const Tensor& output, const Tensor& weights, const Tensor& bias = Tensor(),
                     const Padding& padding = sy_valid, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    _context = context;
    _input = &input;
    _output = &output;
    _weights = &weights;
    _bias = (bias.dim() > 0 ? &bias : nullptr);
    _padding = padding;
    _stride = stride;
    _dilation = dilation;
    // validate here shapes/layouts/types here
  }

  virtual void bind(cl::Context context, const Tensor& input, const Tensor& output, const Tensor& weights,
                     const Padding& padding = sy_valid, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    bind(context, input, output, weights, Tensor(), padding, stride, dilation);
  }

  virtual void compile() {
    std::stringstream opt;
    opt << "-I opencl_kernels ";
    opt << getTensor4DOption("INPUT", *_input);
    opt << getTensor4DOption("FILTER", *_weights);
    opt << getTensor4DOption("OUTPUT", *_output);
    opt << "-D COMPUTE_TYPE=" << "float" << " ";
    opt << "-D STRIDE_Y=" << _stride[0] << " ";
    opt << "-D STRIDE_X=" << _stride[1] << " ";
    opt << "-D DILATION_Y=" << _dilation[0] << " ";
    opt << "-D DILATION_X=" << _dilation[1] << " ";
    opt << "-D INPUT_Y_OFFSET=" << 0 << " ";
    opt << "-D INPUT_X_OFFSET=" << 0 << " ";
    if (_bias) {
      opt << "-D BIAS_TYPE=" << "float" << " ";
      opt << "-D BIAS_TERM=true" << " ";
    }
    compile_kernel(_context, 0, opt.str());
  }

  virtual cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    // validate allocated buffers
    auto k = kernel(0);
    k.setArg(0, _input->buffer());
    k.setArg(1, _output->buffer());
    k.setArg(2, _weights->buffer());
    if (_bias)
      k.setArg(3, _bias->buffer());
    cl::NDRange gws = cl::NDRange(_output->shape(3), _output->shape(2), _output->shape(1) * _output->shape(0));
    cl::NDRange lws = cl::NullRange;
    cl::NDRange offset = cl::NullRange;
    return queue.enqueueNDRangeKernel(kernel(0), offset, gws, lws, events, event);
  }
protected:
  cl::Context _context;
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