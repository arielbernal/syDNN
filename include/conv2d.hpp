#pragma once
#include <fstream>
#include <iostream>
#include <unordered_map>

#include <conv2d_naive.hpp>
#include <conv2d_3x3.hpp>

namespace syDNN {

// struct SyKernel {
//   cl::Kernel kernel;
//   cl::NDRange gws = cl::NullRange;
//   cl::NDRange lws = cl::NullRange;
//   cl::NDRange offset = cl::NullRange;
// };

// inline std::string load_program(const std::string& filename) {
//   std::ifstream ifs(filename.c_str());
//   if (!ifs.is_open())
//     throw std::ios_base::failure("load_program -> File not found: " + filename);

//   std::string ret;
//   ret.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
//   return ret;
// }

// inline SyKernel get_kernel(cl::Context context, const std::string& sourceCode, const std::string& kernelName,
//                   const std::string& compileOptions = "") {
//   SyKernel k;
//   cl_int err = CL_SUCCESS;
//   std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
//   cl::Program program = cl::Program(context, sourceCode, false);
//   err = program.build(compileOptions.c_str());
//   if (err != CL_SUCCESS) {
//     for (auto& e : devices) {
//       std::string device_name = e.getInfo<CL_DEVICE_NAME>();
//       std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(e);
//       std::cout << "Device: " << device_name << std::endl;
//       std::cout << log << std::endl;
//     }
//     throw std::runtime_error("get_kernel -> build error");
//   }
//   k.kernel = cl::Kernel(program, kernelName.c_str());
//   return k;
// }


// std::string getTensorOption(const std::string& name, const Tensor& t)
// {
//   std::stringstream opt;
//   opt << "-D " << name << "_TYPE=" << "float" << " ";
//   opt << "-D " << name << "_B=" << t.shape(0) << " ";
//   opt << "-D " << name << "_F=" << t.shape(1) << " ";
//   opt << "-D " << name << "_Y=" << t.shape(2) << " ";
//   opt << "-D " << name << "_X=" << t.shape(3) << " ";
//   opt << "-D " << name << "_B_PITCH=" << t.pitch(0) << " ";
//   opt << "-D " << name << "_F_PITCH=" << t.pitch(1) << " ";
//   opt << "-D " << name << "_Y_PITCH=" << t.pitch(2) << " ";
//   opt << "-D " << name << "_X_PITCH=" << t.pitch(3) << " ";
//   opt << "-D " << name << "_B_PADDING=" << t.padding(0) << " ";
//   opt << "-D " << name << "_F_PADDING=" << t.padding(1) << " ";
//   opt << "-D " << name << "_Y_PADDING=" << t.padding(2) << " ";
//   opt << "-D " << name << "_X_PADDING=" << t.padding(3) << " ";
//   return opt.str();
// }




// SyKernel conv2d(cl::Context context, const Tensor& input, const Tensor& weights, const Tensor& output,
//                 const Tensor& bias = NullTensor,
//                 const Size& stride = {1, 1}, const Size& dilation = {1, 1})
// {
//   std::string strProgram = load_program("opencl_kernels/conv2d_naive.cl");

//   size_t filter_y_radius = weights.shape(2) / 2;
//   size_t filter_x_radius = weights.shape(3) / 2;
//   size_t input_y_padding = input.padding(2);
//   size_t input_x_padding = input.padding(3);
//   size_t output_b = output.shape(0);
//   size_t output_f = output.shape(1);
//   // size_t output_y = output.shape(2);
//   // size_t output_x = output.shape(3);

//   size_t output_y = 1 + (int(input.shape(2)) + 2 * input_y_padding - ((weights.shape(2) - 1) * dilation[0] + 1)) / stride[0];
//   size_t output_x = 1 + (int(input.shape(3)) + 2 * input_x_padding - ((weights.shape(3) - 1) * dilation[1] + 1)) / stride[1];

//   std::stringstream opt;
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

//   SyKernel k = get_kernel(context, strProgram, "convolution", opt.str());

//   k.kernel.setArg(0, input.buffer());
//   k.kernel.setArg(1, output.buffer());
//   k.kernel.setArg(2, weights.buffer());
//   if (bias.allocated())
//     k.kernel.setArg(3, bias.buffer());

//   k.gws = cl::NDRange(output_x, output_y, output_f * output_b);

//   return k;
// }


// class ImplementationBase {
//   std::string _programFileName;
//   std::string _kernelName;
//   std::string _kernelOptions;
// };


// class Conv2DImplementation : public ImplementationBase {
// public:
//    layout getInputLayout();
//    layout getOutputLayout();
//    layout getWeightsLayout();
//    shape getInputShape();
//    shape getOutputShape();
//    shape getWeightsShape();
// };

// class Conv2DNaive : public Conv2DImplementation {
// public:
//   void getOptions() {
//     std::stringstream opt;
//     opt << "-I opencl_kernels ";
//     opt << getTensorOption("INPUT", input);
//     opt << getTensorOption("FILTER", weights);
//     opt << getTensorOption("OUTPUT", output);
//     opt << "-D COMPUTE_TYPE=" << "float" << " ";
//     opt << "-D STRIDE_Y=" << stride[0] << " ";
//     opt << "-D STRIDE_X=" << stride[1] << " ";
//     opt << "-D DILATION_Y=" << dilation[0] << " ";
//     opt << "-D DILATION_X=" << dilation[1] << " ";
//     opt << "-D INPUT_Y_OFFSET=" << (input_y_padding > filter_y_radius ? input_y_padding - filter_y_radius : 0) << " ";
//     opt << "-D INPUT_X_OFFSET=" << (input_x_padding > filter_x_radius ? input_x_padding - filter_x_radius : 0) << " ";
//     if (bias.allocated()) {
//       opt << "-D BIAS_TYPE=" << "float" << " ";
//       opt << "-D BIAS_TERM=true" << " ";
//     }
//   }
// };

// class Conv2D1x1 : public Conv2DImplementation {
// public:
//   void getOptions() {}
// };

// class Conv2D3x3 : public Conv2DImplementation {
// public:
//   void getOptions() {}
// };



// class Conv2D {
// public:
//   std::vector<Conv2DImplementation> getImplementations() {

//   }
// protected:
//   void load_program() {}
//   void getOptions() {}
//   void compile() {}
//   static register_implementation(Conv2DImplementation& s);
// private:
//   static std::vector<Conv2DImplementation> implementations;
// };


// std::vector<Conv2DImpl> impls = Conv2D::getIplementations();
// Conv2D k (impl, X, W, b, Y);





} // namespace syDNN