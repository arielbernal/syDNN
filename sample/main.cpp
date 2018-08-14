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
  std::cout << "Default Conv2D Implementation : " << Conv2DFactory::default_implementation() << std::endl;
  auto conv_impls = Conv2DFactory::implementations();
  std::cout << "Implementations = " << conv_impls.size() << "\n";
  for (auto &e : conv_impls) {
    std::cout << "  " << e.first << std::endl;
  }

  Tensor X({1, 2, 7, 7}, {0, 0, 1, 1});
  Tensor W({3, 2, 3, 3});
  Tensor b({3}, {0});
  Tensor Y(Conv2DFactory::output_shape(X, W, sy_same));

  // std::string bestImpl = Conv2DFactory::best_implementation(X, W, sy_same);
  // std::vector<ProfilingInfo> profilingInfo = Conv2DFactory::profiling_list(X, W, sy_same);

  std::cout << "Executing " << "Conv2DNaive" << std::endl;
  Conv2DPtr conv2d = Conv2D("Conv2DNaive", clContext, X, Y, W, b, sy_same);
  conv2d->compile();

  X.allocate(clContext);
  W.allocate(clContext);
  b.allocate(clContext);
  Y.allocate(clContext);
  conv2d->set_arguments();

  X.map(clQueue, false);
  W.map(clQueue, false);
  b.map(clQueue);
  X.copy((void*)Xr.data());
  W.copy((void*)Wr.data());
  b.copy((void*)br.data());
  X.unmap(clQueue);
  W.unmap(clQueue);
  b.unmap(clQueue);

  cl_int err = conv2d->enqueue(clQueue);
  clQueue.finish();

  Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
  std::cout << Y.to_string<float>("%4.0f", true) << std::endl;

  std::cout << "Executing " << "Conv2D_bfyx_os_iyx_osv16" << std::endl;
  Tensor Y1(Conv2DFactory::output_shape(X, W, sy_same));
  Conv2DPtr conv2d1 = Conv2D("Conv2D_bfyx_os_iyx_osv16", clContext, X, Y1, W, b, sy_same);
  conv2d1->compile();

  Y1.allocate(clContext);
  conv2d1->set_arguments();

  err = conv2d1->enqueue(clQueue);
  clQueue.finish();

  Y1.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
  std::cout << Y1.to_string<float>("%4.0f", true) << std::endl;


// Graph API-------------------------------------------------------------------------------------------------------
// General ConvNet
// conv2d(graph, input, filters, kernel_size, t0, [stride, padding, dilation, data_format, activation, use_bias]);
// relu(graph, t1, t2);
// conv2d(graph, t2, filters, kernel_size, t3, [stride, padding, dilation, data_format, activation, use_bias]);
// relu(graph, t3, t4);
// maxpooling2d(graph, pool_size, t4, t5, stride, padding, data_format)
// dropout(graph, t5, t6);
// flatten(graph, t6, t7);
// dense(graph, 32, t7, t8);

// Inception Layer
// conv2d(graph, input, filters = 64, kernel_size = {1, 1}, t0, [stride, padding = same, dilation, {data_format, activation = relu, use_bias}]);
// conv2d(graph, t0, filters, kernel_size, t1, [stride, padding, dilation, {data_format, activation = relu, use_bias}]);
// conv2d(graph, input, filters, kernel_size, t2, [stride, padding, dilation, {data_format, activation = relu, use_bias}]);
// conv2d(graph, t2, filters, kernel_size, t3, [stride, padding, dilation, {data_format, activation = relu, use_bias}]);
// maxpooling(graph, pool_size, input, t4, [stride, padding]);
// conv2d(graph, t4, filters, kernel_size, t5, [stride, padding, dilation, {data_format, activation = relu, use_bias}]);
// concatenate(graph, std::vector<Tensor> T6 = {t1, t3, t5}, [axis = 1]);

//---------------------------------------------------------------------------------------------------------------------


  return 0;
}
