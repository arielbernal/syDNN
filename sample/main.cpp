#include <iostream>
#include <map>
#include <functional>

#include <tensor.hpp>
#include <sydnn.hpp>
#include <conv2d.hpp>
#include <tensor_ref.hpp>
#include <test_common.hpp>


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

  TensorRefF Xr = RandomTensorRef({1, 2, 7, 7}, {0, 0, 1, 1}, 1, 10);
  TensorRefF Wr = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
  TensorRefF Yr = TensorRefF({1, 3, 7, 7}, {0, 0, 0, 0});
  TensorRefF br = RandomTensorRef({3}, {0}, 1, 10);

  // conv2d_ref1(Xr, Wr, Yr, br, 2, 2);
  // //conv2d_ref1(Xr, Wr, Yr);
  // std::cout << Yr.to_string("%4.0f") << std::endl;

  Tensor X({1, 2, 7, 7}, {0, 0, 1, 1});
  Tensor W({3, 2, 3, 3});
  Tensor b({3}, {0});
  Tensor Y;

  Conv2D conv2d = Conv2DFactory::create("Conv2DNaive");
  conv2d->bind(clContext, X, Y, W, b, sy_same);
 
  Y.resize(conv2d->output_shape(), {0, 0, 1, 1});

  conv2d->compile();

  X.allocate(clContext);
  X.map(clQueue);
  X.from_buffer((void*)Xr.data());
  X.unmap(clQueue);

  W.allocate(clContext);
  W.map(clQueue);
  W.from_buffer((void*)Wr.data());
  W.unmap(clQueue);

  b.allocate(clContext);
  b.map(clQueue);
  b.from_buffer((void*)br.data());
  b.unmap(clQueue);

  Y.allocate(clContext);


  cl_int err = conv2d->enqueue(clQueue);
  clQueue.finish();


  Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
  std::cout << Y.to_string<float>("%4.0f", true) << std::endl;

// Graph API
  // conv2d(graph, tin, kernel_size, stride, padding, tout);
  // relu(graph, tin);
//----------------------------------------


  return 0;
}
