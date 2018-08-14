#pragma once

#include <utilities.hpp>
#include <conv2d/conv2d_impl.hpp>

namespace syDNN {

#define ROUNDUP(sz, n)      ((sz) + (n) - 1 - (((sz) + (n) - 1) % (n)))

class Conv2D_bfyx_os_iyx_osv16 : public Conv2DBase {
public:
  static const size_t sub_group_size;

  Conv2D_bfyx_os_iyx_osv16() = default;

  Conv2D_bfyx_os_iyx_osv16(const cl::Context& context, const Tensor& input, const Tensor& output,
            const Tensor& weights, const Tensor& bias, const Padding& padding, const Size& stride, const Size& dilation)
  : Conv2DBase(context, input, output, weights, bias, padding, stride, dilation)
  {
    add_kernel("opencl_kernels/conv2d_bfyx_os_iyx_osv16.cl", "convolution_gpu_bfyx_os_iyx_osv16");
  }

  Layout input_layout() override { return sy_nchw; }
  Layout weights_layout() override { return sy_nchw; }
  Layout output_layout() override { return sy_nchw; }

  std::vector<Type> input_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> output_type() override { return {sy_fp32, sy_fp16}; }
  std::vector<Type> weights_type() override { return {sy_fp32, sy_fp16}; }

  void compile() {
    std::cout << "Compile Conv2D_bfyx_os_iyx_osv16\n";

    const int32_t of_maps = _output->shape(1);
    const int32_t output_block_width = sub_group_size - _weights->shape(3) + 1;
    const int32_t output_block_height = 2;
    const int32_t prefecth = 4;

    // Number of elements in X dimension needed from input to compute output block without re-reading input.
    int32_t input_block_req_width = (output_block_width - 1) * _stride[1] + (_weights->shape(3) - 1) * _dilation[1] + 1;
    // Number of elements in Y dimension needed from input to compute output block without re-reading input.
    int32_t input_block_req_height = (output_block_height - 1) * _stride[0] + (_weights->shape(2) - 1) * _dilation[0] + 1;
    int32_t read_chunk_size = sub_group_size / 2; // fp16 used then -> sub_group_size
    int32_t min_read_size = sub_group_size;

    // Required number of elements in X dimension rounded to nearest >= read chunk size.
    int32_t input_block_read_width = std::max(ROUNDUP(input_block_req_width, read_chunk_size), min_read_size);
    // Number of sub-group-sized vectors of unit type needed to store input block.
    int32_t input_block_array_size = std::ceil(input_block_req_height * input_block_read_width / float(sub_group_size));

    std::stringstream preamble;
    preamble << kernel_define("FUNC(x)", "x");
    preamble << kernel_define("FUNC_CALL(x)", "x");
    preamble << kernel_define("KERNEL(x)", "void __kernel x");
    preamble << kernel_define("SUB_GROUP_SIZE", sub_group_size);
    preamble << kernel_define("OUTPUT_BLOCK_HEIGHT", output_block_height);
    preamble << kernel_define("OUTPUT_BLOCK_WIDTH", output_block_width);
    preamble << kernel_define("IN_BLOCK_ARRAY_SIZE", input_block_read_width);
    preamble << kernel_define("IN_BLOCK_WIDTH", input_block_read_width);
    preamble << kernel_define("PREFETCH", prefecth);
    preamble << kernel_define("FILTER_OFM_NUM", _weights->shape(0));
    preamble << kernel_define("FILTER_IFM_NUM", _weights->shape(1));
    preamble << kernel_define("FILTER_SIZE_Y", _weights->shape(2));
    preamble << kernel_define("FILTER_SIZE_X", _weights->shape(3));
    preamble << kernel_define("INPUT0_BATCH_PITCH", _input->pitch(0));
    preamble << kernel_define("INPUT0_FEATURE_PITCH", _input->pitch(1));
    preamble << kernel_define("INPUT0_Y_PITCH", _input->pitch(2));
    preamble << kernel_define("INPUT0_X_PITCH", _input->pitch(3));
    preamble << kernel_define("INPUT0_OFFSET_WITH_PADDING", 0);
    preamble << kernel_define("OUTPUT_BATCH_PITCH", _output->pitch(0));
    preamble << kernel_define("OUTPUT_FEATURE_PITCH", _output->pitch(1));
    preamble << kernel_define("OUTPUT_Y_PITCH", _output->pitch(2));
    preamble << kernel_define("OUTPUT_X_PITCH", _output->pitch(3));
    preamble << kernel_define("OUTPUT_OFFSET", 0);
    preamble << kernel_define("OUTPUT_SIZE_Y", _output->shape(2));
    preamble << kernel_define("OUTPUT_SIZE_X", _output->shape(3));
    preamble << kernel_define("STRIDE_SIZE_Y", _stride[0]);
    preamble << kernel_define("STRIDE_SIZE_X", _stride[1]);
    preamble << kernel_define("DILATION_SIZE_Y", _dilation[0]);
    preamble << kernel_define("DILATION_SIZE_X", _dilation[1]);
    preamble << kernel_loop_unroll_macro();
    if (_bias) {
      preamble << kernel_define("BIAS_TERM", "true");
      preamble << kernel_define("BIAS_TYPE", "float");
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
    size_t leftovers = 0;
    k.add_argument(leftovers);
  }

private:
  static bool _registered;
};

const size_t Conv2D_bfyx_os_iyx_osv16::sub_group_size = 16;

bool Conv2D_bfyx_os_iyx_osv16::_registered = Conv2DFactory::register_implementation<Conv2D_bfyx_os_iyx_osv16>("Conv2D_bfyx_os_iyx_osv16");

} // namespace syDNN
