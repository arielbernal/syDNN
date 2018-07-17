#pragma once

#include <iterator>
#include <iostream>
#include <sydnn.hpp>
#include <size.hpp>
#include <utilities.hpp>

namespace syDNN {

class Tensor {
public:
  Tensor(cl::Context context, const Size& shape, const Size& padding, const Size& alignment, Type type = Type::clrt_fp32)
  : _context(context)
  , _shape(shape)
  , _padding(padding)
  , _alignment(alignment)
  , _internal(align(shape + 2 * padding, alignment))
  , _type(type)
  {
    update_buffer_layout();
  }

  Tensor(cl::Context context, const Size& size, const Size& padding, Type type = Type::clrt_fp32)
  : Tensor(context, size, padding, Size::Fill(size.size(), 1), type) {}

  Tensor(cl::Context context, const Size& size, Type type = Type::clrt_fp32)
  : Tensor(context, size, Size::Zeros(size.size()), Size::Fill(size.size(), 1), type) {}

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "Tensor shape = " << _shape << ", padding = " << _padding << ", alignment = " << _alignment << ", ";
    ret << type_name(_type) << ", ";
    ret << (allocated() ? "allocated" : "not-allocated") << ", ";
    ret << (mapped() ? "mapped" : "unmapped");
    return ret.str();
  }

  Size shape() {
    return _shape;
  }

  Size padding() {
    return _padding;
  }

  Size alignment() {
    return _alignment;
  }

  Size pitch() {
    return _pitch;
  }

  Size internal() {
    return _internal;
  }

  size_t buffer_size() const {
    return _buffer_size;
  }

  bool allocated() const {
    return _allocated;
  }

  bool mapped() const {
    return _mapped_ptr != nullptr;
  }

  template<typename T = void>
  T* mapped_ptr() {
    return static_cast<T*>(_mapped_ptr);
  }

  void allocate(cl_mem_flags flags = CL_MEM_READ_WRITE, void* host_ptr = nullptr, cl_int* err = nullptr) {
    if (_allocated)
      throw std::runtime_error("Tensor::allocate");
    cl_int error;
    _buffer = cl::Buffer(_context, flags, _buffer_size, host_ptr, &error);
    _allocated = (error == CL_SUCCESS);
    if (err != nullptr) *err = error;
  }

  template<typename T = void>
  T* map(cl::CommandQueue q, bool blocking = true, cl_map_flags flags = CL_MAP_WRITE,
              const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr, cl_int* err = nullptr) {
    if (_mapped_ptr != nullptr)
      throw std::runtime_error("Tensor::map");
    cl_int error;
    _mapped_ptr = q.enqueueMapBuffer(_buffer, blocking, flags, 0, _buffer_size, events, event, &error);
    _mapped_flags = flags;
    if(err != nullptr) *err = error;
    return static_cast<T*>(_mapped_ptr);
  }

  cl_int unmap(cl::CommandQueue q, bool blocking = true, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    cl_int err = CL_SUCCESS;
    if (_mapped_ptr != nullptr) {
      err = q.enqueueUnmapMemObject(_buffer, _mapped_ptr, events, event);
      if (blocking) {
        q.finish();
      }
      _mapped_ptr = nullptr;
    }
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
    return *reinterpret_cast<Ret*>(static_cast<uint8_t*>(_mapped_ptr) + idx);
  }

  template <typename Ret>
  Ret& at(const Size& p) const{
    size_t idx = index(p);
    return *reinterpret_cast<Ret*>(static_cast<uint8_t*>(_mapped_ptr) + idx);
  }

protected:
  void update_buffer_layout() {
    if (_internal.size() > 0) {
      _pitch = Size::Zeros(_internal.size());
      _pitch[0] = type_size(_type);
      _buffer_size = type_size(_type) * _internal[0];
      for (int i = 1; i < _internal.size(); ++i) {
        _pitch[i] = _pitch[i - 1] * _internal[i - 1];
        _buffer_size *= _internal[i];
      }
    } else {
      _buffer_size = 0;
    }
  }

private:
  cl::Context _context;
  Size _shape;
  Size _padding;
  Size _alignment;
  Size _internal;
  Type _type;

  Size _pitch;
  size_t _buffer_size;
  bool _allocated = false;
  cl::Buffer _buffer;
  cl_map_flags _mapped_flags = CL_MAP_READ;
  void* _mapped_ptr = nullptr;
};

} // namespace syDNN