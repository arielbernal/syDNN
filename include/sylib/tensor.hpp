#pragma once

#include <iterator>
#include <iostream>
#include <random>
#include <stack>
#include <CL/cl.hpp>
#include <sylib/sylibdef.hpp>
#include <sylib/size.hpp>

namespace sylib {

class Tensor {
public:
  Tensor() : _type(Type::sy_fp32), _dim(0), _allocated(false) {}

  Tensor(const Size& shape, const Size& padding, const Size& alignment, Type type = Type::sy_fp32)
  : _shape(shape)
  , _padding(padding)
  , _alignment(alignment)
  , _internal(align(shape + 2 * padding, alignment))
  , _type(type)
  , _dim(shape.size())
  {
    update_buffer_layout();
  }

  Tensor(const Size& shape, const Size& padding, Type type = Type::sy_fp32)
  : Tensor(shape, padding, Size::Fill(shape.size(), 1), type) {}

  Tensor(const Size& shape, Type type = Type::sy_fp32)
  : Tensor(shape, Size::Zeros(shape.size()), Size::Fill(shape.size(), 1), type) {}

  void resize(const Size& shape, const Size& padding = Size(), const Size& alignment = Size()) {
    if (allocated())
      throw std::runtime_error("Tensor::resize -> resize of allocated tensor");
    _shape = shape;
    _padding = padding.size() == 0 ? Size::Zeros(shape.size()) : padding;
    _alignment = alignment.size() == 0 ? Size::Fill(shape.size(), 1) : alignment;
    _internal = align(_shape + 2 * _padding, _alignment);
    _dim = _shape.size();
    update_buffer_layout();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "Tensor shape = " << _shape << ", padding = " << _padding << ", alignment = " << _alignment << ", ";
    ret << type_name(_type) << ", ";
    ret << (allocated() ? "allocated" : "not-allocated");
    ret << " " << _buffer() << std::endl;
    return ret.str();
  }

  Size shape() const { return _shape; }
  Size padding() const { return _padding; }
  Size alignment() const { return _alignment; }
  Size pitch() const { return _pitch; }
  Size internal() { return _internal; }
  int32_t shape(size_t idx) const { return _shape[idx]; }
  int32_t padding(size_t idx) const { return _padding[idx]; }
  int32_t alignment(size_t idx) const { return _alignment[idx]; }
  int32_t pitch(size_t idx) const { return _pitch[idx]; }
  int32_t internal(size_t idx) const { return _internal[idx]; }
  size_t buffer_size() const { return _buffer_size; }
  size_t dim() const { return _dim; }
  bool allocated() const { return _allocated; }

  // provided for debugging. The map count returned should be considered immediately stale. OpenCL 1.2 spec
  size_t map_count() const {  return allocated() ? _buffer.getInfo<CL_MEM_MAP_COUNT>() : 0; }
  // provided for debugging. The reference count returned should be considered immediately stale. OpenCL 1.2 spec
  size_t ref_count() const {  return allocated() ? _buffer.getInfo<CL_MEM_REFERENCE_COUNT>() : 0; }

  operator bool() const { return _dim > 0; }
  cl::Buffer operator()() const { return _buffer; }
  void operator()(cl::Buffer& buffer) {
    _buffer = buffer;
    _allocated = true;
  }

  template<typename T = void>
  T* mapped_ptr() { return static_cast<T*>(_mapped_ptr); }

  void allocate(cl::Context context, cl_mem_flags flags = CL_MEM_READ_WRITE, void* host_ptr = nullptr, cl_int* err = nullptr) {
    cl_int error;
    _buffer = cl::Buffer(context, flags, _buffer_size, host_ptr, &error);
    _allocated = (error == CL_SUCCESS);
    if (err != nullptr) *err = error;
  }

  template<typename T = void>
  T* map(cl::CommandQueue q, bool blocking = true, cl_map_flags flags = CL_MAP_WRITE,
              const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr, cl_int* err = nullptr) {
    cl_int error;
    _mapped_ptr = static_cast<uint8_t*>(q.enqueueMapBuffer(_buffer, blocking, flags, 0, _buffer_size, events, event, &error));
    _mapped_flags = flags;
    if(err != nullptr) *err = error;
    return reinterpret_cast<T*>(_mapped_ptr);
  }

  cl_int unmap(cl::CommandQueue q, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    cl_int err = CL_SUCCESS;
    err = q.enqueueUnmapMemObject(_buffer, _mapped_ptr, events, event);
    return err;
  }

  template <typename T, typename... Rest>
  size_t index(T t, Rest... rest) const {
    if ((sizeof...(Rest) + 1) != _pitch.size())
      throw std::runtime_error("Tensor::buffer_index");
    T arr[sizeof...(Rest) + 1] = { t, rest...};
    size_t acc = 0;
    for(size_t i = 0; i < _pitch.size(); ++i)
      acc += _pitch[i] * (_padding[i] + arr[i]);
    return acc;
  }

  size_t index(const Size& p) const {
    if (p.size() != _pitch.size())
      throw std::runtime_error("Tensor::buffer_index");
    size_t acc = 0;
    for (int i = 0; i < _pitch.size(); ++i)
      acc += _pitch[i] * (_padding[i] + p[i]);
    return acc;
  }

  template <typename Ret, typename T, typename... Rest>
  Ret& at(T t, Rest... rest) const {
    size_t idx = index(t, rest...);
    return reinterpret_cast<Ret*>(_mapped_ptr)[idx];
  }

  template <typename Ret>
  Ret& at(const Size& p) const{
    size_t idx = index(p);
    return reinterpret_cast<Ret*>(_mapped_ptr)[idx];
  }

  void copy(void* data) {
    if (_mapped_ptr && (_mapped_flags & CL_MAP_WRITE)) {
      std::memcpy(_mapped_ptr, data, _buffer_size);
    }
  }

  template<typename T>
  std::string to_string(const std::string& format, bool internal = false, size_t n = 0, size_t idx = 0) {
    if (n == _dim) {
      char str[100];
      snprintf(str, 100, format.c_str(), reinterpret_cast<T*>(_mapped_ptr)[idx]);
      return std::string(str) + " " ;
    }
    std::string ret;
    if (n == 0) {
      ret = to_string() + "\n";
    }
    for (size_t i = 0; i < (internal? _internal[n] : _shape[n]); ++i) {
      ret = ret + to_string<T>(format, internal, n + 1, idx + (i + (internal ? 0 : _padding[n])) * _pitch[n]);
    }
    return ret + "\n";
  }




protected:
  void update_buffer_layout() {
    if (_internal.size() > 0) {
      _pitch = Size::Zeros(_internal.size());
      _pitch[_dim - 1] = 1;
      _buffer_size = type_size(_type) * _internal[_dim - 1];
      for (int i = _dim - 2 ; i >= 0 ; --i) {
        _pitch[i] = _pitch[i + 1] * _internal[i + 1];
        _buffer_size *= _internal[i];
      }
    } else {
      _buffer_size = 0;
    }
  }
private:
  Tensor& operator=(const Tensor&) = delete;

  Size _shape;
  Size _padding;
  Size _alignment;
  Size _internal;
  Type _type;
  size_t _dim;

  Size _pitch;
  size_t _buffer_size;
  bool _allocated = false;
  cl::Buffer _buffer;
  cl_map_flags _mapped_flags = CL_MAP_READ;
  uint8_t* _mapped_ptr = nullptr;
};

inline void random_values(Tensor& t, int start, int end) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(start, end);
  std::stack<size_t> st;
  int n = t.dim() - 1;
  st.push(0);
  Size shape = t.shape();
  Size p = Size::Zeros(shape.size());
  while(!st.empty()) {
    int i = st.top();
    if (p[i] < shape[i] && i < n) {
      st.push(i + 1);
    } else {
      if (i == n) {
        for (int x = 0; x < shape[i]; ++x) {
          p[i] = x;
          t.at<float>(p) = distribution(generator);
        }
      }
      p[i] = 0;
      st.pop();
      if (!st.empty()) {
        int j = st.top();
        p[j]++;
      }
    }
  };
}

} // namespace sylib