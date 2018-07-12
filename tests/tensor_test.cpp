#include <tensor.hpp>
#include <common/test_common.hpp>

using namespace syDNN;

// TEST(Tensor_test, constructor) {
//   Tensor t(clContext, {1, 2, 3}, {1, 2, 3}, {1, 2, 3});
//   Size shape {1, 2, 3};
//   ASSERT_EQ(shape, t.shape());
// }

// TEST(Tensor_test, constructor_runtime_error) {
//   try {
//     Tensor t(clContext, {1, 2, 3}, {1, 2}, {2, 3, 3});
//     FAIL() << "Expected std::runtime_error";
//   }
//   catch(const std::runtime_error& err) {
//   }
//   catch(...) {
//     FAIL() << "Expected std::runtime_error";
//   }
// }

// TEST(Tensor_test, pitch) {
//   Tensor t0(clContext, {1, 2, 3}, {1, 2, 3}, {1, 2, 3});
//   ASSERT_EQ(t0.pitch(), Size({4, 12, 72}));
//   Tensor t1(clContext, {2, 1, 3}, {2, 1, 3}, {2, 1, 3});
//   ASSERT_EQ(t1.pitch(), Size({4, 24, 72}));
//   Tensor t2(clContext, {2, 3, 1}, {2, 3, 1}, {2, 3, 1});
//   ASSERT_EQ(t2.pitch(), Size({4, 24, 216}));
// }

// TEST(Tensor_test, buffer_size) {
//   Tensor t(clContext, {1, 2, 3}, {1, 2, 3}, {1, 2, 3});
//   ASSERT_EQ(t.buffer_size(), 3 * 6 * 9 * 4);
// }

// TEST(Tensor_test, allocate) {
//   Tensor t(clContext, {3, 3});
//   t.allocate();
//   ASSERT_EQ(t.allocated(), true);
// }

// TEST(Tensor_test, allocate_host_ptr) {
//   Tensor t(clContext, {3, 3});
//   std::vector<float> v(9);
//   t.allocate(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
//   ASSERT_EQ(t.allocated(), true);
// }

// TEST(Tensor_test, map) {
//   Tensor t(clContext, {3, 3});
//   std::vector<float> v(9);
//   t.allocate(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
//   cl_int err = CL_TRUE;
//   void* ptr = t.map(clQueue);
//   ASSERT_EQ((void*)&v[0], ptr);
//   ASSERT_EQ(t.mapped(), true);
// }

// TEST(Tensor_test, unmap) {
//   Tensor t(clContext, {3, 3});
//   std::vector<float> v(9);
//   t.allocate(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &v[0]);
//   float* ptr = t.map<float>(clQueue);
//   ASSERT_EQ(&v[0], ptr);
//   t.unmap(clQueue);
//   ASSERT_EQ(t.mapped(), false);
// }

// TEST(Tensor_test, unmap_write) {
//   Tensor t(clContext, {3, 3});
//   t.allocate(CL_MEM_READ_WRITE);
//   float* ptr1 = t.map<float>(clQueue, true, CL_MAP_WRITE);
//   ASSERT_EQ(t.mapped(), true);
//   ptr1[6] = 33;
//   t.unmap(clQueue);
//   ASSERT_EQ(t.mapped(), false);
//   float* ptr2 = t.map<float>(clQueue);
//   ASSERT_EQ(ptr2[6], 33);
// }