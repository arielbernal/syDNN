#pragma once

#include <dnn/dense/dense_impl.hpp>

namespace sylib {
namespace dnn {

class DenseNaive : public DenseBase {
public:
  DenseNaive() = default;
  DenseNaive(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias)
  : DenseBase(context, input, output, weights, bias)
  {
    add_kernel("opencl_kernels/dense_naive.cl", "dense");
  }

  Layout input_layout() override { return sy_nw; }
  Layout weights_layout() override { return sy_nw; }
  Layout output_layout() override { return sy_nw; }

  std::vector<Type> input_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> output_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> weights_type() override { return {sy_fp32, sy_fp16}; }

  void compile() override {
    std::cout << "DenseNaive::compile\n";
    std::stringstream preamble;
    preamble << kernel_define("FUNC(x)", "x");
    preamble << kernel_define("FUNC_CALL(x)", "x");
    preamble << kernel_define("KERNEL(x)", "void __kernel x");
    preamble << getTensor2DOption("INPUT", _input);
    preamble << getTensor2DOption("FILTER", _weights);
    preamble << getTensor2DOption("OUTPUT", _output);
    preamble << kernel_define("COMPUTE_TYPE", "float");
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
    k.global_work_size(cl::NDRange(_output.shape(1), _output.shape(0)));
    k.add_argument(_input());
    k.add_argument(_output());
    k.add_argument(_weights());
    if (_bias)
      k.add_argument(_bias());
  }

private:
  static bool _registered;
};

bool DenseNaive::_registered = DenseFactory::register_implementation<DenseNaive>("DenseNaive", true);

} // namespace dnn
} // namespace sylib