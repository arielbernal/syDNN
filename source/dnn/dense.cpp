#include <memory>
#include <sylib/dnn/dense.hpp>
#include <dnn/dense/dense_impl.hpp>
#include <dnn/dense/dense_naive.hpp>

namespace sylib {
namespace dnn {

// Operation constructor
Dense::Dense(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias)
: Operation(DenseFactory::create(name, context, input, output, weights, bias))
{}

// Specific Operation functions
Size Dense::output_shape(const Tensor& input, const Tensor& weights)
{
  return {input.shape(0), weights.shape(1)};
}

std::vector<Type> Dense::input_type(const std::string& name)
{
  std::cout << "Here Dense::input_type\n";
  std::shared_ptr<DenseBase> s = DenseFactory::get_instance(name);
  std::cout << "Object = " << &s << " " << s->test() << std::endl;
  return DenseFactory::get_instance(name)->input_type();
}

} // dnn
} // sylib