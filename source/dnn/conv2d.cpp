#include <memory>
#include <sylib/dnn/conv2d.hpp>
#include <dnn/conv2d/conv2d_impl.hpp>
#include <dnn/conv2d/conv2d_naive.hpp>
#include <dnn/conv2d/conv2d_bfyx_os_iyx_osv16.hpp>

namespace sylib {
namespace dnn {

bool Conv2DNaive::_registered =
            Conv2DFactory::register_implementation<Conv2DNaive>("Conv2DNaive", true);
bool Conv2D_bfyx_os_iyx_osv16::_registered =
            Conv2DFactory::register_implementation<Conv2D_bfyx_os_iyx_osv16>("Conv2D_bfyx_os_iyx_osv16", false);

class Conv2D::Conv2DImpl {
public:
  Conv2DImpl(std::unique_ptr<Conv2DBase> ptr) : _ptr(std::move(ptr)) {}
  ~Conv2DImpl() = default;
private:
  std::unique_ptr<Conv2DBase> _ptr;
};

Conv2D::Conv2D(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding,
            const Size& stride, const Size& dilation)
: _impl {std::make_unique<Conv2DImpl>(
            Conv2DFactory::create(name, context, input, output, weights, bias, padding, stride, dilation))}
{}

Conv2D::~Conv2D() = default;

Size Conv2D::output_shape(const Tensor& input, const Tensor& weights,
            const Padding& padding, const Size& stride, const Size& dilation)
{
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

} // dnn
} // sylib