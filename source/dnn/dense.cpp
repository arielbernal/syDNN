#include <memory>
#include <sylib/dnn/dense.hpp>
#include <dnn/dense/dense_impl.hpp>
#include <dnn/dense/dense_naive.hpp>

namespace sylib {
namespace dnn {

Dense::Dense(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias)
: Operation(DenseFactory::create(name, context, input, output, weights, bias))
{}

// General methods
Size Dense::output_shape(const Tensor& input, const Tensor& weights)
{
  return {input.shape(0), weights.shape(1)};
}

// Implementation specific methods
std::vector<Type> Dense::input_type(const std::string& name)
{
  return DenseFactory::get_instance(name)->input_type();
}

} // dnn
} // sylib