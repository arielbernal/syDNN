#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <utilities.hpp>

namespace syDNN
{

using size_value_type = int32_t;
const size_t MAX_SIZE_DIM = 16;

class Size {
public:
  using Array = std::array<size_value_type, MAX_SIZE_DIM>;

  Size() {}

  Size(const Size& rhs)
  : _arr(rhs._arr)
  , _size(rhs._size)
  {}

  Size(const size_value_type* c_arr, size_t n)
  : _size(n)
  {
    std::copy(c_arr, c_arr + _size, std::begin(_arr));
  }

  template<typename... Ts, class = typename std::enable_if<all_integral<Ts...>::value>::type>
  Size(Ts&&... ts)
  : _arr({std::forward<size_value_type>(ts)...})
  , _size(sizeof...(Ts))
  {
    static_assert(all_same<size_value_type, Ts...>::value, "Size elements should be of integral type");
    static_assert(sizeof...(Ts) <= MAX_SIZE_DIM, "Number of dimensions greater than MAX_SIZE_DIM");
  }

  void resize(size_t n, size_value_type v = 0) {
    if (n > MAX_SIZE_DIM)
      throw std::out_of_range("Size::resize");
    _size = n;
    _arr.fill(v);
  }

  void fill(size_value_type v = 0) {
    _arr.fill(v);
  }

  void range(size_value_type n, size_value_type step) {
    for (size_t i = 0; i < _size; ++i)
      _arr[i] = n + step * i;
  }

  static Size Fill(size_t n, size_value_type v = 0) {
    Size ret;
    ret.resize(n, v);
    return ret;
  }

  static Size Zeros(size_t n) {
    Size ret;
    ret.resize(n, 0);
    return ret;
  }

  static Size Range(size_value_type n0, size_value_type n1, size_value_type step = size_value_type(1)) {
    Size ret;
    if (n1 < n0 && step > 0 || n0 < n1 && step < 0 || step == 0)
      return ret;
    size_t n = std::ceil(std::fabs(n1 - n0) / std::fabs(step));
    ret.resize(n, 0);
    ret.range(n0, step);
    return ret;
  }

  static Size Range(size_value_type n) {
    return Range(0, n);
  }

  size_value_type& operator[](size_t n) {
    return _arr[n];
  }

  size_value_type operator[](size_t n) const {
    return _arr[n];
  }

  const size_value_type& at(size_t n) const {
    if (n > _size)
      throw std::out_of_range("Size::at");
    return _arr[n];
  }

  Size& operator=(const Size& rhs) {
    _arr = rhs.data();
    _size = rhs.size();
    return *this;
  }

  void operator=(const size_value_type rhs) {
    _arr.fill(rhs);
  }

  Size add(const Size& rhs) const {
    if (rhs._size != _size)
      throw std::runtime_error("Size::add");
    Size s = *this;
    for (int i = 0; i < _size; ++i)  s[i] += rhs[i];
    return s;
  }

  Size mul(const size_value_type rhs) const {
    Size s = *this;
    for (int i = 0; i < _size; ++i)  s[i] *= rhs;
    return s;
  }

  Size& operator+=(const Size& rhs) {
    if (rhs._size != _size)
      throw std::runtime_error("Size::+=");
    for (size_t i = 0; i < _size; i++)
      _arr[i] += rhs[i];
    return *this;
  }

  Size& operator*=(const size_value_type rhs) {
    for (size_t i = 0; i < _size; i++)
      _arr[i] *= rhs;
    return *this;
  }

  bool operator==(const Size& rhs) const {
    if (rhs._size != _size)
      return false;
    return std::equal(_arr.begin(), _arr.begin() + _size, rhs.data().begin());
  }

  size_value_type dim() const {
    return std::accumulate(_arr.begin(),_arr.begin() + _size, static_cast<size_value_type>(1), std::multiplies<size_t>());
  }

  size_t size() const {
    return _size;
  }

  const Array& data() const {
    return _arr;
  }

  Array data() {
    return _arr;
  }

  friend std::ostream& operator<<(std::ostream& os, const Size& s) {
    os << s.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "[";
    if (_size > 0) {
      for (size_t i = 0; i < _size - 1; ++i)
        ret << _arr[i] << ", ";
      ret << _arr[_size - 1];
    }
    ret << "]";
    return ret.str();
  }

private:
  Array _arr;
  size_t _size = 0;
};

inline Size operator+(const Size& lhs, const Size& rhs) { return lhs.add(rhs); }
inline Size operator*(const Size& lhs, const size_value_type rhs) { return lhs.mul(rhs); }
inline Size operator*(const size_value_type lhs, const Size& rhs) { return rhs.mul(lhs); }

inline Size align(const Size& v, const Size& alignment) {
  if (v.size() != alignment.size())
    throw std::runtime_error("align(Size&, Size&)");
  Size s = Size::Zeros(v.size());
  for (size_t i = 0; i < s.size(); ++i)
    s[i] = syDNN::align(v[i], alignment[i]);
  return s;
}

} // namespace syDNN