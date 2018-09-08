#include <sylib/tensor.hpp>
#include <common/test_common.hpp>

using namespace sylib;

TEST(Tensor_test, constructor) {
  Tensor t({1, 2, 3}, {1, 2, 3}, {1, 2, 3});
  Size shape {1, 2, 3};
  ASSERT_EQ(shape, t.shape());
}

TEST(Tensor_test, constructor_runtime_error) {
  try {
    Tensor t({1, 2, 3}, {1, 2}, {2, 3, 3});
    FAIL() << "Expected std::runtime_error";
  }
  catch(const std::runtime_error& err) {
  }
  catch(...) {
    FAIL() << "Expected std::runtime_error";
  }
}

TEST(Tensor_test, pitch) {
  Tensor t0({1, 2, 3}, {1, 2, 3}, {1, 2, 3});
  ASSERT_EQ(t0.pitch(), Size({54, 9, 1}));
  Tensor t1({2, 1, 3}, {2, 1, 3}, {2, 1, 3});
  ASSERT_EQ(t1.pitch(), Size({27, 9, 1}));
  Tensor t2({2, 3, 1}, {2, 3, 1}, {2, 3, 1});
  ASSERT_EQ(t2.pitch(), Size({27, 3, 1}));
}

TEST(Tensor_test, buffer_size) {
  Tensor t({1, 2, 3}, {1, 2, 3}, {1, 2, 3});
  ASSERT_EQ(t.buffer_size(), 3 * 6 * 9 * 4);
}

TEST(Tensor_test, allocate) {
  Tensor t(Size{3, 3});
  t.allocate(clContext);
  ASSERT_EQ(t.allocated(), true);
}

TEST(Tensor_test, allocate_host_ptr) {
  Tensor t(Size{3, 3});
  std::vector<float> v(9);
  t.allocate(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
  ASSERT_EQ(t.allocated(), true);
}

TEST(Tensor_test, map) {
  Tensor t(Size{3, 3});
  std::vector<float> v(9);
  t.allocate(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
  cl_int err = CL_TRUE;
  void* ptr = t.map(clQueue);
  ASSERT_EQ((void*)&v[0], ptr);
  ASSERT_EQ(t.map_count(), 1);
}

TEST(Tensor_test, unmap) {
  Tensor t(Size{3, 3});
  std::vector<float> v(9);
  t.allocate(clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
  float* ptr = t.map<float>(clQueue);
  ASSERT_EQ(&v[0], ptr);
  t.unmap(clQueue);
  ASSERT_EQ(t.map_count(), 0);
}

TEST(Tensor_test, unmap_write) {
  Tensor t(Size{3, 3});
  t.allocate(clContext, CL_MEM_READ_WRITE);
  float* ptr1 = t.map<float>(clQueue, true, CL_MAP_WRITE);
  ASSERT_EQ(t.map_count(), 1);
  ptr1[6] = 33;
  t.unmap(clQueue);
  ASSERT_EQ(t.map_count(), 0);
  float* ptr2 = t.map<float>(clQueue);
  ASSERT_EQ(ptr2[6], 33);
}


TEST(Tensor_test, copy_contructor) {
  Tensor t(Size{3, 3});
  t.allocate(clContext, CL_MEM_READ_WRITE);
  float* ptr1 = t.map<float>(clQueue, true, CL_MAP_WRITE);
  ptr1[6] = 33;
  t.unmap(clQueue);
  float* ptr2 = t.map<float>(clQueue);
  ASSERT_EQ(ptr2[6], 33);
  ASSERT_EQ(ptr1, ptr2);
  t.map_count();
  Tensor f = t;
  t.map_count();
  f.map_count();
  t.unmap(clQueue);
  std::cout << "T = " << t << std::endl;
  std::cout << "F = " << f << std::endl;
  t.map_count();
  f.map_count();
  // cl::Buffer b(clContext, CL_MEM_READ_WRITE, _buffer_size, host_ptr, &error);


}