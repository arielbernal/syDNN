// #include <sylib/syop.hpp>
// namespace sylib {

// template<class TBase>
// Operation<TBase>::Operation(const std::unique_ptr<TBase> ptr): _ptr(std::move(ptr)) {}

// template<class TBase>
// Operation<TBase>::~Operation() { }

// template<class TBase>
// void Operation<TBase>::compile() { _ptr->compile(); }

// template<class TBase>
// void Operation<TBase>::set_arguments() { _ptr->set_arguments(); }

// template<class TBase>
// cl_int Operation<TBase>::enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events, cl::Event* event) {
//   _ptr->enqueue(queue, events, event);
// }

// } // namespace sylib