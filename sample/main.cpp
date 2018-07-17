#include <iostream>
#include <tensor.hpp>
#include <conv2d.hpp>

using half_t = uint16_t;



int main() {
  using namespace syDNN;

  std::string platform_name = "Intel";
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  std::vector<cl::Device> all_gpu_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_gpu_devices);
  if(all_gpu_devices.size() == 0)
    throw std::runtime_error("TestEnvironment() : No gpu devices found");

  cl::Device default_device = all_gpu_devices[0];
  std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
  cl::Context clContext = cl::Context({default_device});
  cl::CommandQueue clQueue = cl::CommandQueue(clContext, default_device);

  std::vector<cl::Device>ctxs =  clContext.getInfo<CL_CONTEXT_DEVICES>();
  for (auto& c : ctxs) {
     std::cout << c.getInfo<CL_DEVICE_NAME>() << std::endl;
  }

  Tensor X(clContext, {12, 12, 1, 1});
  Tensor W(clContext, {3, 3, 3});
  Tensor b(clContext, {3, 3, 3});


  Kernel k = conv_2d(clContext, tin, W, b, stride, tout);
//----------------------------------------
// API design
//----------------------------------------
  // Full control
  // Create program, compile, create kernel, set_args.
  // Convolution2DInfo info = get_info<conv2d>(Input_shape, Weights_shape, stride, padding);
  // Kernel k = conv_2d(clContext, tin, W, b, stride, tout);
  // k.enqueue(clQueue); // or  clQueue.enqueueNDRange(k.kernel, k.offset, k.glogal, k.local, events, event)

// Graph API
  // conv2d(graph, tin, kernel_size, stride, padding, tout);
  // relu(graph, tin);
//----------------------------------------


  return 0;
}