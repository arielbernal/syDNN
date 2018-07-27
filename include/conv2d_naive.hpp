#pragma once

#include <conv2d_impl.hpp>

namespace syDNN {

class Conv2DNaiveImpl : public Conv2DImplBase {
public:
  Conv2DNaiveImpl()
  : Conv2DImplBase("opencl_kernels/conv2d_naive.cl")
  {
    add_kernel("convolution");
  }

  std::string get_options() {
   std::stringstream opt;
//   opt << "-I opencl_kernels ";
//   opt << getTensorOption("INPUT", input);
//   opt << getTensorOption("FILTER", weights);
//   opt << getTensorOption("OUTPUT", output);
//   opt << "-D COMPUTE_TYPE=" << "float" << " ";
//   opt << "-D STRIDE_Y=" << stride[0] << " ";
//   opt << "-D STRIDE_X=" << stride[1] << " ";
//   opt << "-D DILATION_Y=" << dilation[0] << " ";
//   opt << "-D DILATION_X=" << dilation[1] << " ";
//   opt << "-D INPUT_Y_OFFSET=" << (input_y_padding > filter_y_radius ? input_y_padding - filter_y_radius : 0) << " ";
//   opt << "-D INPUT_X_OFFSET=" << (input_x_padding > filter_x_radius ? input_x_padding - filter_x_radius : 0) << " ";
//   if (bias.allocated()) {
//     opt << "-D BIAS_TYPE=" << "float" << " ";
//     opt << "-D BIAS_TERM=true" << " ";
//   }
    return opt.str();
  }
private:
  static bool _registered;
};


bool Conv2DNaiveImpl::_registered = Conv2DFactory::register_implementation("Conv2DNaive", true,
                           []() -> std::unique_ptr<Conv2DImplBase> {return std::make_unique<Conv2DNaiveImpl>();});


}