#include <memory>
#include <sylib/syop.hpp>
#include <implementation.hpp>

namespace sylib {

Operation::Operation(ImplPtr impl) : _impl(std::move(impl))
{}

Operation::~Operation() = default;

void Operation::compile()
{
  _impl->compile();
}

void Operation::set_arguments()
{
  _impl->set_arguments();
}

cl_int Operation::enqueue(cl::CommandQueue queue, const std::vector<cl::Event>* events, cl::Event* event)
{
  _impl->enqueue(queue, events, event);
}

} // sylib