#include <iostream>
#include <tensor.hpp>
#include <conv2d.hpp>
#include <tensor_ref.hpp>
#include <test_common.hpp>


void foo(const cl::Device& d = cl::Device()) {

}

int main() {
  using namespace syDNN;
  using namespace syDNN::test;


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

  TensorRefF Xr = RandomTensorRef({1, 2, 5, 5}, {0, 0, 1, 1}, 1, 10);
  TensorRefF Wr = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
  TensorRefF Yr = TensorRefF({1, 3, 5, 5}, {0, 0, 1, 1});
  TensorRefF br = RandomTensorRef({3}, {0}, 1, 10);

  conv2d_ref1(Xr, Wr, Yr, br, 2, 2);
  //conv2d_ref1(Xr, Wr, Yr);
  std::cout << Yr.to_string("%4.0f") << std::endl;


  Tensor X(clContext, {1, 2, 5, 5}, {0, 0, 1, 1});
  Tensor W(clContext, {3, 2, 3, 3});
  Tensor b(clContext, {3});
  Tensor Y(clContext, {1, 3, 3, 3}, {0, 0, 1, 1});


  b.allocate();
  b.map(clQueue);
  b.from_buffer((void*)br.data());
  b.unmap(clQueue);


  X.allocate();
  X.map(clQueue);
  X.from_buffer((void*)Xr.data());
  X.unmap(clQueue);

  W.allocate();
  W.map(clQueue);
  W.from_buffer((void*)Wr.data());
  W.unmap(clQueue);

  Y.allocate();

  //SyKernel k = conv2d(clContext, X, W, Y, NullTensor, {1, 1}, {1, 1});
  //SyKernel k = conv2d(clContext, X, W, Y, b);
  SyKernel k = conv2d(clContext, X, W, Y, b, {2, 2});
  cl_int err = clQueue.enqueueNDRangeKernel(k.kernel, k.offset, k.gws, k.lws);
  std::cout << "Kernel execution = " << OpenCLErrorString(err) << std::endl;

  Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
  std::cout << "Output buffer mapped = " << OpenCLErrorString(err) << std::endl;
  std::cout << Y.to_string<float>("%4.0f", true) << std::endl;


  //Kernel k = conv_2d(clContext, tin, W, b, stride, tout);
//----------------------------------------
// API design
//----------------------------------------
  // Full control
  // Create program, compile, create kernel, set_args.
  // Convolution2DInfo info = get_info<conv2d>(Input_shape, Weights_shape, stride, padding);
  // Kernel k = conv2d(clContext, tin, W, b, stride, tout);
  // k.enqueue(clQueue); // or  clQueue.enqueueNDRange(k.kernel, k.offset, k.glogal, k.local, events, event)

// Graph API
  // conv2d(graph, tin, kernel_size, stride, padding, tout);
  // relu(graph, tin);
//----------------------------------------


  return 0;
}