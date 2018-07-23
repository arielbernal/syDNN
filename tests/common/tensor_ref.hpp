#pragma once

#include <size.hpp>
#include <stack>
#include <random>

namespace syDNN {

namespace test {

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
  size_t dim() { return _N; }
  size_t shape(size_t idx) { return _shape[idx]; }
  size_t padding(size_t idx) { return _padding[idx]; }
  size_t pitch(size_t idx) { return _pitch[idx]; }
  size_t internal(size_t idx) { return _internal[idx]; }

  T* data() { return _data; }

  friend std::ostream& operator<<(std::ostream& os, const TensorRef& tensor) {
    os << tensor.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "TensorRef shape = " << _shape << ", padding = " << _padding << ", internal = " << _internal << ", buffer_size = " << _buffer_size;
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



// TODO: remove
template<typename T>
void setValues(int n, TensorRef<T>& t, Size& p) {
  if (n == t.shape().size()) {
    t(p) = 33;
  }
  for (size_t i = 0; i < t.shape()[n]; ++i) {
    p[n] = i;
    setValues(n + 1, t, p);
  }
}


template<typename T>
void setValuesLinear(TensorRef<T>& t, int start, int end) {
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
          t(p) = distribution(generator);
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


template<typename T = float>
inline TensorRef<T> RandomTensorRef(const Size& shape, const Size& padding, int start, int end)
{
  TensorRef<T> ret(shape, padding);
  setValuesLinear(ret, start, end);
  return ret;
}

using TensorRefF = TensorRef<float>;

} // namespace test

} // namespace syDNN