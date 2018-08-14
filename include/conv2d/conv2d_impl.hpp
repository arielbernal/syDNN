#pragma once

#include <implementation.hpp>

namespace syDNN {

class Conv2DBase : public Implementation {
public:
  Conv2DBase() = default;
  Conv2DBase(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding, const Size& stride, const Size& dilation)
  : Implementation(context)
  , _input(&input)
  , _output(&output)
  , _weights(&weights)
  , _bias(&bias)
  , _padding(padding)
  , _stride(stride)
  , _dilation(dilation)
  {}

  virtual Layout input_layout() = 0;
  virtual Layout weights_layout() = 0;
  virtual Layout output_layout() = 0;

  virtual std::vector<Type> input_type() = 0;
  virtual std::vector<Type> output_type() = 0;
  virtual std::vector<Type> weights_type() = 0;

  virtual Size output_shape() = 0;
  virtual void compile() = 0;
  virtual void set_arguments() = 0;

  virtual Size output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) = 0;


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


using Conv2DPtr = std::unique_ptr<Conv2DBase>;
using Conv2DConstructor = std::function<Conv2DPtr(const cl::Context& context, const Tensor& input, const Tensor& output,
                                  const Tensor& weights, const Tensor& bias,
                                  const Padding& padding, const Size& stride, const Size& dilation)>;

class Conv2DFactory : public FactoryBase<Conv2DBase, Conv2DConstructor> {
public:
  using ObjectMap1 = std::unordered_map<std::string, Conv2DPtr>;

  template<class T>
  static bool register_implementation(const std::string& name) {
    return register_impl<T>(name, true, [](const cl::Context& context, const Tensor& input, const Tensor& output,
                      const Tensor& weights, const Tensor& bias,
                      const Padding& padding, const Size& stride, const Size& dilation) -> Conv2DPtr
      { return std::make_unique<T>(context, input, output, weights, bias, padding, stride, dilation); } );
  }
  static Size output_shape(const std::string& name, const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride = {1, 1}, const Size& dilation = {1, 1}) {
    return FactoryBase::get_instance(name)->output_shape(input, weights, padding, stride, dilation);
  }
private:
  Conv2DFactory(Conv2DFactory const&) = delete;
  void operator=(Conv2DFactory const&) = delete;
};


Conv2DPtr Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1}){
  return Conv2DFactory::create(name, context, input, output, weights, bias, padding, stride, dilation);
}

Conv2DPtr Conv2D(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding = sy_valid,
            const Size& stride = {1, 1}, const Size& dilation = {1, 1}){
  return Conv2D("", context, input, output, weights, bias, padding, stride, dilation);
}



}