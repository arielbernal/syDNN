#pragma once

#include <unordered_map>
#include <size.hpp>

namespace syDNN {

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
  sy_nhwc
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

} // namespace syDNN