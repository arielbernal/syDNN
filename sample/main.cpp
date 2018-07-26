#include <iostream>
#include <map>
#include <functional>
#include <tensor.hpp>
#include <conv2d.hpp>
#include <tensor_ref.hpp>
#include <test_common.hpp>


class ImplementationBase {
public:
  std::string _programFileName;
  std::string _kernelName;
  std::string _kernelOptions;

  std::string load_program() {
    std::ifstream ifs(_programFileName.c_str());
    if (!ifs.is_open())
      throw std::ios_base::failure("load_program -> File not found: " + _programFileName);

    std::string ret;
    ret.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return ret;
  }
};


class Conv2DImplBase : public ImplementationBase {
public:
   virtual void get_options(int x, int y) = 0;
   virtual Size output_shape(const Size& input_shape, const Size& weights_shape) = 0;
};

class Conv2DNaiveImpl : public Conv2DImplBase {
public:
  Conv2DNaiveImpl() {
    std::cout << "Constructor Conv2DNaiveImpl\n";
    std::cout << "Loading program\n";
  }
  void get_options(int x, int y) { std::cout << "Conv2DNaiveImpl->get_options\n"; }
  void Size output_shape(const Size& input_shape, const Size& weights_shape, const Size& ) {
      size_t output_y = 1 + (int(input_shape(2)) + 2 * input_y_padding - ((weights.shape(2) - 1) * dilation[0] + 1)) / stride[0];
      size_t output_x = 1 + (int(input_shape(3)) + 2 * input_x_padding - ((weights.shape(3) - 1) * dilation[1] + 1)) / stride[1];

  }
private:
  static bool _registered;
};

class Conv2D {
public:
  using CreateImplementation = std::function<std::unique_ptr<Conv2DImplBase>()>;

  static std::unique_ptr<Conv2DImplBase> create_implementation(const std::string& name) {
    auto it = _implementations.find(name);
    if (it != _implementations.end())
      return it->second();

    return nullptr;
  }

  static bool register_implementation(const std::string name, CreateImplementation funcCreate) {
    auto it = _implementations.find(name);
    if (it == _implementations.end()) {
      _implementations[name] = funcCreate;
      return true;
    }
    return false;
  }
private:
  static std::map<std::string, CreateImplementation> _implementations;
};

std::map<std::string, Conv2D::CreateImplementation> Conv2D::_implementations;

bool Conv2DNaiveImpl::_registered = Conv2D::register_implementation("Conv2DNaive",
                           []() -> std::unique_ptr<Conv2DImplBase> {return std::make_unique<Conv2DNaiveImpl>();});

using Conv2DImpl = std::unique_ptr<Conv2DImplBase>;

int main() {

  Conv2DImpl impl = Conv2D::create_implementation("Conv2DNaive");
  impl->get_options(1, 2);




  // using namespace syDNN;
  // using namespace syDNN::test;


  // std::string platform_name = "Intel";
  // std::vector<cl::Platform> all_platforms;
  // cl::Platform::get(&all_platforms);

  // cl::Platform default_platform = all_platforms[0];
  // std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  // std::vector<cl::Device> all_gpu_devices;
  // default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_gpu_devices);
  // if(all_gpu_devices.size() == 0)
  //   throw std::runtime_error("TestEnvironment() : No gpu devices found");

  // cl::Device default_device = all_gpu_devices[0];
  // std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
  // cl::Context clContext = cl::Context({default_device});
  // cl::CommandQueue clQueue = cl::CommandQueue(clContext, default_device);

  // std::vector<cl::Device>ctxs =  clContext.getInfo<CL_CONTEXT_DEVICES>();
  // for (auto& c : ctxs) {
  //    std::cout << c.getInfo<CL_DEVICE_NAME>() << std::endl;
  // }

  // TensorRefF Xr = RandomTensorRef({1, 2, 5, 5}, {0, 0, 1, 1}, 1, 10);
  // TensorRefF Wr = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
  // TensorRefF Yr = TensorRefF({1, 3, 5, 5}, {0, 0, 1, 1});
  // TensorRefF br = RandomTensorRef({3}, {0}, 1, 10);

  // conv2d_ref1(Xr, Wr, Yr, br, 2, 2);
  // //conv2d_ref1(Xr, Wr, Yr);
  // std::cout << Yr.to_string("%4.0f") << std::endl;


  // Tensor X(clContext, {1, 2, 5, 5}, {0, 0, 1, 1});
  // Tensor W(clContext, {3, 2, 3, 3});
  // Tensor b(clContext, {3});
  // Tensor Y(clContext, {1, 3, 3, 3}, {0, 0, 1, 1});


  // b.allocate();
  // b.map(clQueue);
  // b.from_buffer((void*)br.data());
  // b.unmap(clQueue);


  // X.allocate();
  // X.map(clQueue);
  // X.from_buffer((void*)Xr.data());
  // X.unmap(clQueue);

  // W.allocate();
  // W.map(clQueue);
  // W.from_buffer((void*)Wr.data());
  // W.unmap(clQueue);

  // Y.allocate();

  // //SyKernel k = conv2d(clContext, X, W, Y, NullTensor, {1, 1}, {1, 1});
  // //SyKernel k = conv2d(clContext, X, W, Y, b);
  
  // SyKernel k = conv2d(clContext, X, W, Y, b, {2, 2});
  // cl_int err = clQueue.enqueueNDRangeKernel(k.kernel, k.offset, k.gws, k.lws);
  // std::cout << "Kernel execution = " << OpenCLErrorString(err) << std::endl;

  // Y.map<float>(clQueue, true, CL_MAP_READ, nullptr, nullptr, &err);
  // std::cout << "Output buffer mapped = " << OpenCLErrorString(err) << std::endl;
  // std::cout << Y.to_string<float>("%4.0f", true) << std::endl;


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