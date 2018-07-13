#include <iostream>
#include <tensor.hpp>


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

  Tensor t(clContext, {3, 4}, {1,1});
  t.allocate(CL_MEM_READ_WRITE);
  t.map(clQueue, true, CL_MAP_WRITE);
  std::cout << "t = " << t << " " << t.pitch() << std::endl;

  std::cout << "Value at = " << t.at<float>(2, 3) << std::endl;
  t.at<float>(2, 3) = 4;
  std::cout << "Value at = " << t.at<float>(2, 3) << std::endl;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 3; ++i)
      t.at<float>(i, j) = i + j * 3;

  t.unmap(clQueue);
  t.map(clQueue);

  for (int j = -1; j < 5; ++j) {
    for (int i = -1; i < 4; ++i) {
      std::cout << t.at<float>(i, j) << " ";
    }
    std::cout << "\n";
  }

  // Full control
  Program program = Program(context, source);
  program.build(devices);
  Kernel kernel(program, "vector_add");

  struct DNNKernel {
    cl::Kernel kernel;
    cl::Range global;
    cl::Range local;
    cl::Range offset;
  }

  // Create program, compile, create kernel, set_args.
  getInfo<conv2d>(tin.shape(), W.shape(), stride, padding);
  DNNKernel k = conv2d(clContext, tin, W, b, stride, tout);

  k.enqueue(clQueue); // or 
  clQueue.enqueueNDRange(k.kernel, k.offset, k.glogal, k.local, events, event)

  conv2d(graph, tin, kernel_size, stride, padding, tout);
  relu(graph, tin);



  return 0;
}