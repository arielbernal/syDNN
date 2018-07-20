#include <iostream>
#include <tensor.hpp>
#include <conv2d.hpp>
#include <tensor_ref.hpp>


using half_t = uint16_t;

using namespace syDNN::test;


template<typename T>
TensorRef<T> conv2d_ref(TensorRef<T>& input, TensorRef<T>& filter, TensorRef<T> bias, int stride_x = 1, int stride_y = 1)
{
  size_t input_batch = input.shape(0);
  size_t input_channel = input.shape(1);
  size_t input_y = input.shape(2);
  size_t input_x = input.shape(3);

}
//   size_t batch_size = input.size();
//   size_t feature_size = input[0].size();
//   size_t input_y = input[0][0].size() - 2 * input_padding_y;
//   size_t input_x = input[0][0][0].size() - 2 * input_padding_x;
//   size_t bias_size = bias.size();

//   size_t output_filter_size = filter.size();
//   size_t input_filter_size = filter[0].size();
//   size_t filter_y = filter[0][0].size();
//   size_t filter_x = filter[0][0][0].size();
//   size_t offset_x = input_padding_x > (filter_x / 2) ? input_padding_x - filter_x / 2 : 0;
//   size_t offset_y = input_padding_y > (filter_y / 2) ? input_padding_y - filter_y / 2 : 0;

//   size_t output_y = 1 + (int(input_y) - filter_y) / stride_y + 2 * output_padding_y;
//   size_t output_x = 1 + (int(input_x) - filter_x) / stride_x + 2 * output_padding_x;

//   if (input_filter_size != feature_size)
//     std::cout << "Error invalid filter size\n";

//   std::cout << "input  = [" << batch_size << ", " << feature_size << ", " << input_x << ", " << input_y << "]\n";
//   std::cout << "filter = [" << output_filter_size << ", " << input_filter_size << ", " << filter_x << ", " << filter_y << "]\n";
//   std::cout << "output = [" << batch_size << ", " << output_filter_size << ", " << output_x << ", " << output_y << "]\n";
//   std::cout << "bias = [" << bias_size << "]\n";

//   vec4df out = vec4df(batch_size, vec3df(output_filter_size, vec2df(output_y, vec1df(output_x, 0))));

//   for (size_t b = 0; b < batch_size; ++b) {
//     for (size_t fo = 0; fo < output_filter_size; ++fo) {
//       for (size_t y = 0; y < output_y - 2 * output_padding_y; ++y) {
//         for (size_t x = 0; x < output_x - 2 * output_padding_x; ++x) {
//           T acc = 0;
//           for (size_t fi = 0; fi < input_filter_size; ++fi) {
//             for (size_t fy = 0; fy < filter_y; ++fy) {
//               for (size_t fx = 0; fx < filter_x; ++fx) {
//                 int ix = int(stride_x) * x + fx + offset_x;
//                 int iy = int(stride_y) * y + fy + offset_y;
//                 if (ix < 0 || iy < 0) continue;
//                 acc += input[b][fi][iy][ix] * filter[fo][fi][fy][fx];
//               } // fx
//             } // fy
//           } // fi
//           out[b][fo][y + output_padding_y][x + output_padding_x] = acc + bias[fo];
//         } // x
//       } // y
//     } // fo
//   } // b
//   return out;
// }




int main() {
  TensorRefF input = RandomTensorRef({1, 2, 5, 5}, {0, 0, 0, 0}, 1, 10);
  std::cout << input << std::endl;
  TensorRefF filter = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
  std::cout << filter << std::endl;


  return 0;
}



// int main() {
//   using namespace syDNN;

//   vec4df in0 = createVec4df(2, 2, 7, 7, 1, 1);
//   vec4df filter0 = createVec4df(3, 2, 3, 3);
//   vec1df bias = { 1, 1, 1 };
//   print(in0, "Input");
//   print(filter0, "Filter");
//   vec4df out0 = conv2d_ref(in0, filter0, bias, 1, 1, 1, 1);
//   print(out0, "Output");
//   vec4df out1 = conv2d_ref(in0, filter0, bias, 1, 1);
//   print(out1, "Output");
//   vec4df out2 = conv2d_ref(in0, filter0, bias, 1, 1, 1, 1, 1, 1);
//   print(out2, "Output");

//   // std::string platform_name = "Intel";
//   // std::vector<cl::Platform> all_platforms;
//   // cl::Platform::get(&all_platforms);

//   // cl::Platform default_platform = all_platforms[0];
//   // std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

//   // std::vector<cl::Device> all_gpu_devices;
//   // default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_gpu_devices);
//   // if(all_gpu_devices.size() == 0)
//   //   throw std::runtime_error("TestEnvironment() : No gpu devices found");

//   // cl::Device default_device = all_gpu_devices[0];
//   // std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
//   // cl::Context clContext = cl::Context({default_device});
//   // cl::CommandQueue clQueue = cl::CommandQueue(clContext, default_device);

//   // std::vector<cl::Device>ctxs =  clContext.getInfo<CL_CONTEXT_DEVICES>();
//   // for (auto& c : ctxs) {
//   //    std::cout << c.getInfo<CL_DEVICE_NAME>() << std::endl;
//   // }

//   // Tensor X(clContext, {12, 12, 1, 1});
//   // Tensor W(clContext, {3, 3, 3});
//   // Tensor b(clContext, {3, 3, 3});


//   //Kernel k = conv_2d(clContext, tin, W, b, stride, tout);
// //----------------------------------------
// // API design
// //----------------------------------------
//   // Full control
//   // Create program, compile, create kernel, set_args.
//   // Convolution2DInfo info = get_info<conv2d>(Input_shape, Weights_shape, stride, padding);
//   // Kernel k = conv_2d(clContext, tin, W, b, stride, tout);
//   // k.enqueue(clQueue); // or  clQueue.enqueueNDRange(k.kernel, k.offset, k.glogal, k.local, events, event)

// // Graph API
//   // conv2d(graph, tin, kernel_size, stride, padding, tout);
//   // relu(graph, tin);
// //----------------------------------------


//   return 0;
// }