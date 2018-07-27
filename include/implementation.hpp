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

class ImplementationBase {
public:
  ImplementationBase(const std::string& programFilename)
  : _programFileName(programFilename)
  , _programSource(load_program(programFilename))
  {}
  std::string kernel_name(size_t idx = 0) { return _kernelNames[idx]; }
  cl::Kernel kernel(size_t idx = 0) { return _kernels[idx]; }
  void add_kernel(const std::string& name) { _kernelNames[_N++] = name;}
  void compile_kernel(const cl::Context& context, size_t kernel_id, const std::string& options) {
    _kernels[kernel_id] = syDNN::compile_kernel(context, _programSource, _kernelNames[kernel_id], options);
  }
private:
  size_t _N = 0;
  std::string _programFileName;
  std::string _programSource;
  std::array<std::string, SY_MAX_KERNELS> _kernelNames;
  std::array<cl::Kernel, SY_MAX_KERNELS> _kernels;
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
  using CreateImplementation = std::function<std::unique_ptr<T>()>;
  using Implementation = SingletonImpl<CreateImplementation>;

  static std::unique_ptr<T> create(const std::string& name) {
    Implementation& impl = Implementation::instance();
    auto it = impl.find(name);
    if (it == impl.end())
      throw std::runtime_error("ImplementationFactory::create");
    return it->second();
  }

  static std::unique_ptr<T> create() {
    return ImplementationFactory::create(Implementation::instance().get_default());
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
  std::stringstream opt;
  opt << "-D " << name << "_TYPE=" << "float" << " ";
  opt << "-D " << name << "_B=" << t.shape(0) << " ";
  opt << "-D " << name << "_F=" << t.shape(1) << " ";
  opt << "-D " << name << "_Y=" << t.shape(2) << " ";
  opt << "-D " << name << "_X=" << t.shape(3) << " ";
  opt << "-D " << name << "_B_PITCH=" << t.pitch(0) << " ";
  opt << "-D " << name << "_F_PITCH=" << t.pitch(1) << " ";
  opt << "-D " << name << "_Y_PITCH=" << t.pitch(2) << " ";
  opt << "-D " << name << "_X_PITCH=" << t.pitch(3) << " ";
  opt << "-D " << name << "_B_PADDING=" << t.padding(0) << " ";
  opt << "-D " << name << "_F_PADDING=" << t.padding(1) << " ";
  opt << "-D " << name << "_Y_PADDING=" << t.padding(2) << " ";
  opt << "-D " << name << "_X_PADDING=" << t.padding(3) << " ";
  return opt.str();
}
} // namespace syDNN