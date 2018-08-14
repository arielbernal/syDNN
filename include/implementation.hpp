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

  cl::Kernel operator()() { return _kernel; }
  cl_int enqueue(cl::CommandQueue& queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    return queue.enqueueNDRangeKernel(_kernel, _global_work_offset, _global_work_size, _local_work_size, events, event);
  }
  void compile(const std::string& preamble = "", const std::string& options = "") {
    _kernel = syDNN::compile_kernel(_context, preamble + _source, _name, options);
  }

  template<typename T>
  void add_argument(const T arg) { _kernel.setArg(_idx++, arg); }

  void global_work_size(const cl::NDRange& range) { _global_work_size = range; }
  void global_work_offset(const cl::NDRange& range) { _global_work_offset = range; }
  void local_work_size(const cl::NDRange& range) { _local_work_size = range; }
private:
  size_t _idx = 0;
  cl::Context _context;
  std::string _name;
  std::string _source;
  cl::Kernel _kernel;
  cl::NDRange _global_work_size;
  cl::NDRange _local_work_size;
  cl::NDRange _global_work_offset;
};

class Implementation {
public:
  Implementation() = default;
  Implementation(const cl::Context context) : _context(context) {}
  Kernel& kernel(size_t kernel_id = 0) { return _kernels[kernel_id]; }
  void add_kernel(const std::string& name, const std::string& sourceFilename) { _kernels.emplace_back(_context, name, sourceFilename); }
  cl::Context context() { return _context; }

  virtual void compile() = 0;
  virtual void set_arguments() = 0;
  virtual cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) {
    Kernel& k = kernel();
    return k.enqueue(queue, events, event);
  }
private:
  cl::Context _context;
  std::vector<Kernel> _kernels;
};

template<class T, typename ConstructorFunc>
class FactoryBase {
public:
  using RegistryMap = std::unordered_map<std::string, ConstructorFunc>;

  static const RegistryMap& implementations() { return registry(); }
  static const std::string& default_implementation() { return default_name(); }

  template<typename... Args>
  static std::unique_ptr<T> create(const std::string& name, Args&&... args) {
    auto it = registry().find(name != "" ? name : default_name());
    if (it == registry().end())
      throw std::runtime_error("ImplementationFactory::create");
    return it->second(std::forward<Args>(args)...);
  }
protected:
  static bool register_impl(const std::string& name, bool default_impl, ConstructorFunc constructorFunc) {
    auto it = registry().find(name);
    if (it == registry().end()) {
      registry()[name] = constructorFunc;
      if (default_impl)
        default_name() = name;
      return true;
    }
    return false;
  }
private:
  FactoryBase() {};
  FactoryBase(FactoryBase const&) = delete;
  void operator=(FactoryBase const&) = delete;

  static std::string& default_name() {
    static std::string impl;
    return impl;
  }

  static RegistryMap& registry() {
    static RegistryMap impl;
    return impl;
  }

};


} // namespace syDNN