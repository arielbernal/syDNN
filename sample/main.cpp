#include <iostream>
#include <map>
#include <functional>

#include <sylib/dnn/conv2d.hpp>
#include <sylib/dnn/dense.hpp>
#include <tensor_ref.hpp>
#include <test_common.hpp>

int main() {
 // int* p = malloc(20);
  int foo = 0;
  void *t = malloc(10);
  using namespace sylib;
  using namespace sylib::dnn;
  using namespace sylib::test;

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

  {
    TensorRefF Xr = RandomTensorRef({1, 2, 7, 7}, {0, 0, 1, 1}, 1, 10);
    TensorRefF Wr = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
    TensorRefF Yr = TensorRefF({1, 3, 7, 7}, {0, 0, 0, 0});
    TensorRefF br = RandomTensorRef({3}, {0}, 1, 10);

    Tensor X({1, 2, 7, 7}, {0, 0, 1, 1}); // bfyx
    Tensor W({3, 2, 3, 3});
    Tensor b({3}, {0});
    Tensor Y(Conv2D::output_shape(X, W, sy_same));

    X.allocate(clContext);
    W.allocate(clContext);
    b.allocate(clContext);
    Y.allocate(clContext);

    std::cout << "Executing " << "Conv2DNaive" << std::endl;
    Conv2D conv2d("Conv2DNaive", clContext, X, Y, W, b, sy_same);
    conv2d.compile();

    conv2d.set_arguments();

    X.map(clQueue, false);
    W.map(clQueue, false);
    b.map(clQueue);
    X.copy((void*)Xr.data());
    W.copy((void*)Wr.data());
    b.copy((void*)br.data());
    X.unmap(clQueue);
    W.unmap(clQueue);
    b.unmap(clQueue);

    cl_int err = conv2d.enqueue(clQueue);
    clQueue.finish();

    Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
    std::cout << Y.to_string<float>("%4.0f", true) << std::endl;
  }
  {
    TensorRefF Xr = RandomTensorRef({1, 3}, {0, 0}, 1, 10);
    TensorRefF Wr = RandomTensorRef({3, 4}, {0, 0}, 1, 10);
    TensorRefF Yr = TensorRefF({1, 4}, {0, 0});
    TensorRefF br = RandomTensorRef({4}, {0}, 1, 10);

    Tensor X(Size{1, 3}); // yx
    Tensor W(Size{3, 4}); // yx
    Tensor b(Size{4});
    Tensor Y(Size{1, 4}); // yx

    std::cout << "Output shape = " << Dense::output_shape(X, W) << std::endl;
    for (auto& e : Dense::input_type("DenseNaive"))
      std::cout << "Type = " << e << std::endl;

    X.allocate(clContext);
    W.allocate(clContext);
    b.allocate(clContext);
    Y.allocate(clContext);

    X.map(clQueue, false);
    W.map(clQueue, false);
    b.map(clQueue);
    X.copy((void*)Xr.data());
    W.copy((void*)Wr.data());
    b.copy((void*)br.data());
    X.unmap(clQueue);
    W.unmap(clQueue);
    b.unmap(clQueue);

    Dense dense("DenseNaive", clContext, X, Y, W, b);
    dense.compile();
    dense.set_arguments();
    cl_int err = dense.enqueue(clQueue);
    clQueue.finish();

    X.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
    std::cout <<"X = " << X.to_string<float>("%4.0f", true) << std::endl;
    W.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
    std::cout <<"W = " << W.to_string<float>("%4.0f", true) << std::endl;
    b.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
    std::cout <<"b = " << b.to_string<float>("%4.0f", true) << std::endl;
    Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
    std::cout <<"Y = " << Y.to_string<float>("%4.0f", true) << std::endl;
  }

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
