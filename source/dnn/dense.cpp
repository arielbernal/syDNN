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
    Size ret {input.shape(0), weights.shape(0), 1, 1};
    return ret;
}

} // dnn
} // sylib