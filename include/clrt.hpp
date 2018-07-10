#pragma once

#include <unordered_map>
#include <size.hpp>

namespace clRT {

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

enum Layout:int32_t {
  clrt_bfyx, // default format
  clrt_bfxy,
  clrt_fbyx,
  clrt_bfx,
  clrt_x
};

static Size layout_order(const Layout& layout)
{
  static const std::unordered_map<int32_t, Size> layout_map {
    {clrt_bfyx, {0, 1, 2, 3}}, // default format in reverse order
    {clrt_bfxy, {1, 0, 2, 3}},
    {clrt_fbyx, {0, 1, 3, 2}},
    {clrt_bfx,  {0, 1, 2}},
  };
  return layout_map.at(layout);
}

} // namespace clRT