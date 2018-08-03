#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <functional>
#include <memory>
#include <utilities.hpp>

namespace syDNN {

const size_t SY_MAX_KERNELS = 5;

class Kernel {
public:
  Kernel(const cl::Context& context, const std::string& sourceFilename, const std::string& name)
  : _context(context)
  , _name(name)
  , _source(load_program(sourceFilename))
  , _global_work_size(cl::NullRange)
  , _global_work_offset(cl::NullRange)
  , _local_work_size(cl::NullRange)
  {}

  cl::Kernel ocl_kernel() { return _kernel; }
  cl_int enqueue(cl::CommandQueue& queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    return queue.enqueueNDRangeKernel(_kernel, _global_work_offset, _global_work_size, _local_work_size, events, event);
  }
  void compile(const std::string& preamble = "", const std::string& options = "") {
    _kernel = syDNN::compile_kernel(_context, preamble + _source, _name, options);
  }

  template<typename T> void add_argument(const T arg) { _kernel.setArg(idx++, arg); }
  void global_work_size(const cl::NDRange& range) { _global_work_size = range; }
  void global_work_offset(const cl::NDRange& range) { _global_work_offset = range; }
  void local_work_size(const cl::NDRange& range) { _local_work_size = range; }
private:
  size_t idx = 0;
  cl::Context _context;
  std::string _name;
  std::string _source;
  cl::Kernel _kernel;
  cl::NDRange _global_work_size;
  cl::NDRange _local_work_size;
  cl::NDRange _global_work_offset;
};


class ImplementationBase {
public:
  ImplementationBase(const cl::Context context) : _context(context) {}
  Kernel& kernel(size_t kernel_id = 0) { 
    return _kernels[kernel_id];
  }
  void add_kernel(const std::string& name, const std::string& sourceFilename) { _kernels.emplace_back(_context, name, sourceFilename); }
  virtual void compile() = 0;
  virtual void set_arguments() = 0;
private:
  cl::Context _context;
  std::vector<Kernel> _kernels;
};


template<typename T>
class SingletonImpl {
public:
  static SingletonImpl &instance() {
    static SingletonImpl instance_;
    return instance_;
  }
  auto end() { return _implementations.end(); }
  void insert(const std::string& name, T func) {
    _implementations[name] = func;
  }
  auto find(const std::string& name) {
    return _implementations.find(name);
  }
  void set_default(const std::string& name) {
    _default = name;
  }
  std::string get_default() { return _default; }
  std::map<std::string, T> implementations() { return _implementations; }
private:
  SingletonImpl() {};
  SingletonImpl(SingletonImpl const&) = delete;
  void operator=(SingletonImpl const&) = delete;
  std::map<std::string, T> _implementations;
  std::string _default;
};

template<class T>
class ImplementationFactory {
public:
  using CreateImplementation = std::function<std::unique_ptr<T>(const cl::Context& context)>;
  using Implementation = SingletonImpl<CreateImplementation>;

  static std::unique_ptr<T> create(const cl::Context& context, const std::string& name) {
    Implementation& impl = Implementation::instance();
    auto it = impl.find(name);
    if (it == impl.end())
      throw std::runtime_error("ImplementationFactory::create");
    return it->second(context);
  }

  static std::unique_ptr<T> create() {
    return ImplementationFactory::create(Implementation::instance().get_default());
  }

  static std::map<std::string, CreateImplementation> implementations() {
    return Implementation::instance().implementations();
  }

  static std::string default_implementation() {
    return Implementation::instance().get_default();
  }

  static bool register_implementation(const std::string& name, bool default_implementation, CreateImplementation funcCreate) {
    Implementation& impl = Implementation::instance();
    auto it = impl.find(name);
    if (it == impl.end()) {
      impl.insert(name, funcCreate);
      if (default_implementation)
         impl.set_default(name);
      return true;
    }
    return false;
  }
};

std::string getTensor4DOption(const std::string& name, const Tensor& t)
{
  std::stringstream ss;
  ss << kernel_define(name + "_TYPE", "float");
  ss << kernel_define(name + "_B", t.shape(0));
  ss << kernel_define(name + "_F", t.shape(1));
  ss << kernel_define(name + "_Y", t.shape(2));
  ss << kernel_define(name + "_X", t.shape(3));
  ss << kernel_define(name + "_B_PITCH", t.pitch(0));
  ss << kernel_define(name + "_F_PITCH", t.pitch(1));
  ss << kernel_define(name + "_Y_PITCH", t.pitch(2));
  ss << kernel_define(name + "_X_PITCH", t.pitch(3));
  ss << kernel_define(name + "_B_PADDING", t.padding(0));
  ss << kernel_define(name + "_F_PADDING", t.padding(1));
  ss << kernel_define(name + "_Y_PADDING", t.padding(2));
  ss << kernel_define(name + "_X_PADDING", t.padding(3));
  return ss.str();
}

} // namespace syDNN