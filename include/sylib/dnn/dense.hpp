// #pragma once

// #include <memory>
// #include <sylib/tensor.hpp>
// #include <sylib/size.hpp>

// namespace sylib {
// namespace dnn {

// class DenseDBase;

// class Dense {
// public:
//   Dense(const std::string& name, const cl::Context& context, const Tensor& input, const Tensor& output,
//             const Tensor& weights, const Tensor& bias);

//   Dense(const cl::Context& context, const Tensor& input, const Tensor& output,
//             const Tensor& weights, const Tensor& bias);
//   ~Dense();
//   void compile();
//   void set_arguments();
//   cl_int enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr);
// private:
//   std::unique_ptr<DenseBase> _ptr;
// };

// } // dnn
// } // sylib