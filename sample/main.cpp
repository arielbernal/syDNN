#include <iostream>
#include <tensor.hpp>
#include <conv2d.hpp>
#include <typeinfo> 

using half_t = uint16_t;

template<typename T> using vec1d = std::vector<T>;
template<typename T> using vec2d = std::vector<vec1d<T>>;
template<typename T> using vec3d = std::vector<vec2d<T>>;
template<typename T> using vec4d = std::vector<vec3d<T>>;

using vec1df = vec1d<float>;
using vec2df = vec2d<float>;
using vec3df = vec3d<float>;
using vec4df = vec4d<float>;


using namespace syDNN;

template<typename T>
class TensorRef {
public:
  TensorRef(const Size& shape, const Size& padding)
  : _shape(shape)
  , _padding(padding)
  , _internal(shape + 2 * padding)
  , _N(shape.size())
  {
    update_buffer_layout();
    _data = new T[_buffer_size];
    std::memset(_data, 0, _buffer_size * sizeof(T));
  }

  ~TensorRef() {
    delete _data;
  }

  size_t buffer_size() { return _buffer_size; }

  Size shape() { return _shape; }

  Size padding() { return _padding; }

  Size internal() { return _internal; }

  Size pitch() { return _pitch; }

  T* data() { return _data; }

  friend std::ostream& operator<<(std::ostream& os, const TensorRef& tensor) {
    os << tensor.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "TensorRef shape = " << _shape << ", padding = " << _padding << ", buffer_size = " << _buffer_size;
    return ret.str();
  }


  std::string to_string1d(size_t idx, const Size& size, const std::string& format) const {
    std::stringstream ret;
    char str[100];
    for (size_t x = 0; x < size[0]; ++x) {
      snprintf(str, 100, format.c_str(), _data[idx + x ]);
      ret << str << " ";
    }
    if (size[0] < _internal[_N - 1])
      ret << "...";  
    return ret.str();
  }

  std::string to_string2d(size_t idx, const Size& pitch, const Size& size, const std::string& format) const{
    std::stringstream ret;
    for (size_t h = 0; h < size[0]; ++h)
      ret << to_string1d(idx + h * pitch[0], size.sub_right(1), format) << "\n";
    if (size[0] < _internal[_N - 2])
      ret << "...\n";
    return ret.str();
  }

  std::string to_string3d(size_t idx, const Size& pitch, const Size& size, const std::string& format) const{
    std::stringstream ret;
    for (size_t c = 0; c < size[0]; ++c)
      ret << to_string2d(idx + c * pitch[0], pitch.sub_right(1), size.sub_right(1), format) << "\n\n";
    if (size[0] < _internal[_N - 3])
      ret << "...\n\n";
    return ret.str();
  }

  std::string to_string4d(size_t idx, const Size& pitch, const Size& size, const std::string& format) const{
    std::stringstream ret;
    for (size_t n = 0; n < size[0]; ++n)
      ret << to_string3d(idx + n * pitch[0], pitch.sub_right(1), size.sub_right(1), format) << "\n\n";
    if (size[0] < _internal[_N - 4])
      ret << "...\n\n";
    return ret.str();
  }

  std::string to_string(const Size& size, const std::string& format) const {
    std::stringstream ret;
    ret << to_string() << "\n";
    int n = _shape.size();
    
    if (n == 4) { // assume NCHW
      ret << to_string4d(0, _pitch, size, format);
    }
    else if(n == 3) {
      ret << to_string3d(0, _pitch, size, format);
    }
    else if(n == 2) {
      ret << to_string2d(0, _pitch, size, format);
    }
    else if(n == 1) {
      ret << to_string1d(0, size, format);
    }

    return ret.str();
  }

  std::string to_string(const std::string& format) const {
    return to_string(_internal, format);
  }

  template <typename K, typename... Rest>
  size_t index(K t, Rest... rest) const {
    if ((sizeof...(Rest) + 1) != _pitch.size())
      throw std::runtime_error("TensorRef::buffer_index");
    K arr[sizeof...(Rest) + 1] = { t, rest...};
    size_t acc = 0;
    for(size_t i = 0; i < _pitch.size(); ++i)
      acc += _pitch[i] * (_padding[i] + arr[i]);
    return acc;
  }

  size_t index(const Size& p) const {
    if (p.size() != _pitch.size())
      throw std::runtime_error("TensorRef::index");
    size_t acc = 0;
    for (int i = 0; i < _pitch.size(); ++i)
      acc += _pitch[i] * (_padding[i] + p[i]);
    return acc;
  }

  template <typename K, typename... Rest>
  T& operator()(K t, Rest... rest) const {
    size_t idx = index(t, rest...);
    return *((_data) + idx);
  }

  template <typename Ret>
  T& operator()(const Size& p) const{
    size_t idx = index(p);
    return *((_data) + idx);
  }
protected:
    void update_buffer_layout() {
    if (_internal.size() > 0) {
      _pitch = Size::Zeros(_internal.size());
      _pitch[_N - 1] = 1;
      _buffer_size = _internal[_N - 1];
      for (int i = _N - 2 ; i >= 0 ; --i) {
        _pitch[i] = _pitch[i + 1] * _internal[i + 1];
        _buffer_size *= _internal[i];
      }
    } else {
      _buffer_size = 0;
    }
  }

private:
  T* _data;
  Size _shape;
  Size _padding;
  Size _internal;
  Size _pitch;
  size_t _buffer_size;
  size_t _N;
};

using TensorRefF = TensorRef<float>;

int main() {
//   TensorRefF t({2, 3, 3});

//   std::cout << t << std::endl;
//   std::cout << t.pitch() << std::endl;
//   std::cout << t.internal() << std::endl;
//   //t(0, 0, 0, 0) = 30;
//   // t(0, 0, 1, 1) = 10;
//   // t(0, 0, 2, 2) = 40;
//   // t(1, 1, 2, 2) = 40;
//   // std::cout << t(0, 0, 0, 0) << std::endl;
// //  std::cout << t.to_string("%4.0f") << "\n";


  TensorRefF t1 ({2, 2, 3, 3}, Size::Zeros(4));
  t1(0, 1, 1, 1) = 10;
  std::cout << t1 << std::endl;
  std::cout << t1.pitch() << std::endl;
  std::cout << t1.internal() << std::endl;
  std::cout << t1.to_string("%4.0f") << "\n";


  return 0;
}


// template<typename T>
// vec4d<T> conv2d_ref(vec4d<T>& input, vec4d<T>& filter, vec1d<T> bias, int stride_x = 1, int stride_y = 1,
//                 int input_padding_x = 0, int input_padding_y = 0, int output_padding_x = 0, int output_padding_y = 0)
// {
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



// vec4df fillVec4df(size_t batch, size_t channel, size_t height, size_t width, float value = 0) {
//    return vec4df(batch, vec3df(channel, vec2df(height, vec1df(width, value))));
//  }

// vec4df createVec4df(size_t batch, size_t channel, size_t height, size_t width, size_t padding_x = 0, size_t padding_y = 0) {
//   vec4df v = fillVec4df(batch, channel, height, width);
//   for (size_t b = 0; b < batch; ++b) {
//     for (size_t c = 0; c < channel; ++c) {
//       int i = 1;
//       for (size_t y = padding_y; y < height - padding_y; ++y) {
//         for (size_t x = padding_x; x < width - padding_x; ++x) {
//           v[b][c][y][x] = i++;
//         }
//       }
//     }
//   }
//   return v;
// }


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