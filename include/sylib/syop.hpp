// #pragma once

// #include <memory>
// #include <CL/cl.hpp>

// namespace sylib {

// template<class TBase>
// class Operation {
// public:
//   Operation(const std::unique_ptr<TBase> ptr);
//   ~Operation();
//   void compile();
//   void set_arguments();
//   cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr);
// protected:
//   std::unique_ptr<TBase> _ptr;
// };

// } // sylib