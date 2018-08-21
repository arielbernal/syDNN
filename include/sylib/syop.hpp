#pragma once

#include <CL/cl.hpp>

namespace sylib {

class Operation {
public:
  virtual void compile() = 0;
  virtual void set_arguments() = 0;
  virtual cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) = 0;
};

} // namespace sylib