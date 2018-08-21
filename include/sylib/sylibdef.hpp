#pragma once

#include <unordered_map>

namespace sylib {

enum Type : int32_t {
  sy_fp16,
  sy_fp32,
  sy_bfloat,
  sy_int8,
  sy_int16,
};

enum Padding : int32_t {
  sy_valid,
  sy_same
};

enum Layout : int32_t {
  sy_nchw,
  sy_nhwc,
  sy_nw
};


static size_t type_size(const Type& type)
{
  static const std::unordered_map<int32_t, size_t> type_map {
    {sy_fp16, 2},
    {sy_fp32, 4},
    {sy_bfloat, 2},
    {sy_int8, 1},
    {sy_int16, 2}
  };
  return type_map.at(type);
}

static std::string type_name(const Type& type)
{
  static const std::unordered_map<int32_t, std::string> type_map {
    {sy_fp16, "fp16"},
    {sy_fp32, "fp32"},
    {sy_bfloat, "bfloat"},
    {sy_int8, "int8"},
    {sy_int16, "int16"}
  };
  return type_map.at(type);
}


inline size_t align(size_t val, size_t alignment)
{
    size_t diff = val % alignment;
    return (diff == 0) ? val : val + alignment - diff;
}

inline size_t pad(size_t val, size_t alignment)
{
    size_t diff = val % alignment;
    return (diff == 0) ? 0 : alignment - diff;
}

inline bool is_aligned(size_t val, size_t alignment)
{
  return !(val % alignment);
}

template <bool...> struct bool_pack;
template <bool... bs>
using all_true = std::is_same<bool_pack<true, bs...>, bool_pack<bs...,true>>;

template <typename T, typename... Ts>
using all_same = all_true<std::is_same<T, Ts>::value...>;

template <typename... Ts>
using all_integral = all_true<std::is_integral<Ts>::value...>;

template <bool B, typename T = void>
struct disable_if { typedef T type; };
template <typename T>
struct disable_if<true, T> {};

} // namespace sylib