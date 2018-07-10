#include <iostream>
#include <tensor.hpp>

int main() {
  using namespace clRT;

  Size s {1, 2, 3};
  s.range(-4, 2);
  std::cout << s << std::endl;

  // std::cout << type_size(clrt_fp16) << std::endl;
  // std::cout << layout_order(clrt_bfyx) << std::endl;
  // uint32_t a = 3;
  // Tensor t(a, {1, 2, 3}, {1, 1, 1}, {2, 2, 2});
  // std::cout << t << std::endl;
  // Tensor t1(a, {1, 2, 3}, {1, 2, 3});
  // std::cout << t1 << std::endl;
  // Tensor t2(a, {1, 2, 3});
  // std::cout << t2 << std::endl;
  // t.get_pitch();

  // 

  // clRT::Tensor t0(context, {10, 20});
  // clRT::Tensor t1(context, {10, 20}, {2, 2});
  // clRT::Tensor t2(context, {10, 20}, {2, 2}, {8, 8});
  // std::cout << t0 << "\n" << t1 << "\n" << t2 << "\n";

  // t0.allocate();
  // t1.allocate();
  // t2.allocate();
  // std::cout << t0 << "\n" << t1 << "\n" << t2 << "\n";

  // void* host_ptr0 = t0.map(queue, true);
  // void* host_ptr1 = t1.map(queue, true);
  // void* host_ptr2 = t2.map(queue, true);
  // std::cout << t0 << "\n" << t1 << "\n" << t2 << "\n";
  // printf("%p %p %p\n", host_ptr0, host_ptr1, host_ptr2);

  // std::cout << t0.get_pitch() << "\n";

  return 0;
}