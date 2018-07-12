#pragma once

#include <unordered_map>
#include <size.hpp>

namespace syDNN {

enum Type : int32_t {
  clrt_fp16,
  clrt_fp32,
  clrt_bfloat,
  clrt_int8,
  clrt_int16,
};

static size_t type_size(const Type& type)
{
  static const std::unordered_map<int32_t, size_t> type_map {
    {clrt_fp16, 2},
    {clrt_fp32, 4},
    {clrt_bfloat, 2},
    {clrt_int8, 1},
    {clrt_int16, 2}
  };
  return type_map.at(type);
}

static std::string type_name(const Type& type)
{
  static const std::unordered_map<int32_t, std::string> type_map {
    {clrt_fp16, "fp16"},
    {clrt_fp32, "fp32"},
    {clrt_bfloat, "bfloat"},
    {clrt_int8, "int8"},
    {clrt_int16, "int16"}
  };
  return type_map.at(type);
}

} // namespace syDNN