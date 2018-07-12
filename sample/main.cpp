#include <iostream>
#include <tensor.hpp>


using half_t = uint16_t;

int main() {


  using namespace clRT;

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

  Tensor t(clContext, {3, 6});
  t.allocate();
  float* ptr = t.map<float>(clQueue);
  std::cout << t << " " << ptr << std::endl;
  Size pitch = t.pitch();
  std::cout << pitch << "   " << t(2, 3) << " " << t.index({2, 3}) << std::endl;
  std::cout << "Value at = " << t<float>(2, 3) << std::endl;
  Size s {2, 3};
  std::cout << "Value at " << s << " = " << t.at<float>(s) << std::endl;


  return 0;
}