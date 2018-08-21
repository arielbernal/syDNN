#pragma once

#include <CL/cl.hpp>

namespace sylib {

class Implementation;

class Operation {
public:
  using ImplPtr = std::unique_ptr<Implementation>;
  Operation(ImplPtr impl);
  ~Operation();
  void compile();
  void set_arguments();
  cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr);
private:
  ImplPtr _impl;
};

} // namespace sylib