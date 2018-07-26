#pragma once

#include <vector>
#include <iostream>
#include <map>
#include <functional>
#include <memory>

namespace syDNN {

class ImplementationBase {
public:
  ImplementationBase(const std::string& programFilename)
  : _programFileName(programFilename)
  , _programSource(load_program(programFilename))
  {}
  std::string programFilename() { return _programFileName; }
  std::string programSource() { return _programSource; }
  std::string kernelName(size_t idx = 0) { return _kernelNames[idx]; }
  std::string kernelOption(size_t idx = 0) { return _kernelOptions[idx]; }
  std::vector<std::string>& kernelNames() { return _kernelNames; }
  std::vector<std::string>& kernelOptions() { return _kernelOptions; }
  void addKernel(const std::string& name) { _kernelNames.push_back(name); }
  void addKernelOptions(const std::string& options) { _kernelOptions.push_back(options); }
private:  
  std::string load_program(const std::string& filename) {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open())
      throw std::ios_base::failure("load_program -> File not found: " + filename);

    std::string ret;
    ret.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return ret;
  }
  std::string _programFileName;
  std::string _programSource;  
  std::vector<std::string> _kernelNames;
  std::vector<std::string> _kernelOptions;
};


template<class T>
class ImplementationFactory {
public:
  using CreateImplementation = std::function<std::unique_ptr<T>()>;
  
  static std::unique_ptr<T> create(const std::string& name) {
    auto it = _implementations.find(name);
    if (it != _implementations.end())
      return it->second();
    throw std::runtime_error("ImplementationFactory::create");
    return nullptr;
  }

  static std::unique_ptr<T> create() {
    return ImplementationFactory::create(_default);
  }

  static std::string default_implementation() { return _default; }

  static bool register_implementation(const std::string& name, bool default_implementation, CreateImplementation funcCreate) {
    auto it = _implementations.find(name);
    if (it == _implementations.end()) {
      _implementations[name] = funcCreate;
      if (default_implementation)
         _default = name;
      return true;
    }
    return false;
  }
private:
  static std::map<std::string, CreateImplementation> _implementations;
  static std::string _default;
};


} // namespace syDNN