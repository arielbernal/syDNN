#pragma once

#include <implementation.hpp>

namespace sylib {
namespace dnn {

class DenseBase : public Implementation {
public:
  DenseBase() = default;
  ~DenseBase() = default;
  DenseBase(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias)
  : Implementation(context)
  , _input(input)
  , _output(output)
  , _weights(weights)
  , _bias(bias)
  {}

  virtual Layout input_layout() = 0;
  virtual Layout weights_layout() = 0;
  virtual Layout output_layout() = 0;

  virtual std::vector<Type> input_type() = 0;
  virtual std::vector<Type> output_type() = 0;
  virtual std::vector<Type> weights_type() = 0;

  virtual void compile() = 0;
  virtual void set_arguments() = 0;
protected:
  const Tensor _input;
  const Tensor _output;
  const Tensor _weights;
  const Tensor _bias;
  Size _input_offset;
  Size _output_offset;
};

using DensePtr = std::unique_ptr<DenseBase>;
using DenseConstructor = std::function<DensePtr(const cl::Context& context, const Tensor& input, const Tensor& output,
                                  const Tensor& weights, const Tensor& bias)>;

class DenseFactory : public FactoryBase<DenseBase, DenseConstructor> {
public:
  template<class T>
  static bool register_implementation(const std::string& name, bool default_impl = false) {
    return register_impl<T>(name, default_impl, [](const cl::Context& context, const Tensor& input, const Tensor& output,
                      const Tensor& weights, const Tensor& bias) -> DensePtr
      { return std::make_unique<T>(context, input, output, weights, bias); } );
  }
private:
  DenseFactory(DenseFactory const&) = delete;
  void operator=(DenseFactory const&) = delete;
};

} // namesapce dnn
} // namespace sylib