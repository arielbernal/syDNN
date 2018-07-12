#include <size.hpp>
#include <common/test_common.hpp>

using namespace syDNN;

TEST(Size_test, initialization_list) {
  int32_t a[] = {1, 2, 3};

  Size y{1, 2, 3};
  ASSERT_EQ(y.size(), 3);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, variadic_initialization) {
  int32_t a[] = {1, 2, 3, 4};

  Size y(1, 2, 3, 4);
  ASSERT_EQ(y.size(), 4);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, from_c_array) {
  int32_t a[] = {1, 2, 3, 4, 5};

  Size y(a, 5);
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, copy_constructor) {
  int32_t a[] = {1, 2, 3, 4, 5};
  Size x(a, 5);
  Size y(x);
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, copy_constructor_assignment) {
  int32_t a[] = {1, 2, 3, 4, 5};
  Size x(a, 5);
  Size y = x;
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, equal_operator) {
  int32_t a[] = {1, 2, 3, 4, 5};
  Size x(a, 5);
  Size y(1, 2);
  y = x;
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, equal_constant_operator) {
  int32_t a[] = {5, 5, 5, 5, 5};
  Size y(1, 2, 3, 4, 6);
  y = 5;
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, accessor_operator) {
  int32_t a[] = {1, 2, 8, 4, 5};
  Size y {1,2,3,4,5};
  y[2] = 8;
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y[i], a[i]);
}

TEST(Size_test, at) {
  int32_t a[] = {1, 2, 8, 4, 5};
  Size y {1, 2, 8, 4, 5};
  ASSERT_EQ(y.size(), 5);
  for (size_t i = 0; i < y.size(); ++i)
    EXPECT_EQ(y.at(i), a[i]);
}

TEST(Size_test, add_method) {
  int32_t a[] = {3, 5, 7};
  Size y {1, 2, 3};
  Size x {2, 3, 4};
  Size z = y.add(x);
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, mul_method) {
  int32_t a[] = {2, 4, 6};
  Size y {1, 2, 3};
  Size z = y.mul(2);
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, add) {
  int32_t a[] = {3, 5, 7};
  Size y {1, 2, 3};
  Size x {2, 3, 4};
  Size z = y + x;
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, mul) {
  int32_t a[] = {2, 4, 6};
  Size y {1, 2, 3};
  Size z = y * 2;
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, mul_swap) {
  int32_t a[] = {2, 4, 6};
  Size y {1, 2, 3};
  Size z = 2 * y;
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, operator_assign_add) {
  int32_t a[] = {3, 5, 7};
  Size y {1, 2, 3};
  Size z {2, 3, 4};
  z += y;
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, operator_assign_mul) {
  int32_t a[] = {4, 6, 8};
  Size z {2, 3, 4};
  z *= 2;
  ASSERT_EQ(z.size(), 3);
  for (size_t i = 0; i < z.size(); ++i)
    EXPECT_EQ(z[i], a[i]);
}

TEST(Size_test, dim) {
  Size z {2, 3, 4};
  ASSERT_EQ(z.dim(), 24);
}

TEST(Size_test, data_accessor) {
  int32_t a[] = {2, 3, 4};
  Size z {2, 3, 4};
  const Size::Array x = z.data();
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(x[i], a[i]);
}

TEST(Size_test, data_mutator) {
  int32_t a[] = {10, 3, 4};
  Size z {2, 3, 4};
  auto x = z.data();
  x[0] = 10;
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(x[i], a[i]);
}

TEST(Size_test, fill_zeros) {
  Size z = Size::Zeros(3);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], 0);
}

TEST(Size_test, fill_constant) {
  Size z = Size::Fill(3, -101);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], -101);
}

TEST(Size_test, resize) {
  Size z {1, 2, 3, 4};
  z.resize(3, -101);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], -101);
}

TEST(Size_test, fill) {
  Size z {1, 2, 3, 4};
  z.fill(-101);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], -101);
}

TEST(Size_test, Fill) {
  Size z = Size::Fill(10, -101);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], -101);
}

TEST(Size_test, Zeros) {
  Size z = Size::Zeros(10);
  for (size_t i = 0; i < z.size(); ++i)
    ASSERT_EQ(z[i], 0);
}

TEST(Size_test, comparison) {
  Size z {1, 2, 3, 4};
  Size a {1, 2, 5, 4};
  Size b {1, 2, 3};
  Size c {1, 2, 3, 4};
  bool comparison = (a == z);
  ASSERT_EQ(comparison, false);
  comparison = (z == b);
  ASSERT_EQ(comparison, false);
  comparison = (z == c);
  ASSERT_EQ(comparison, true);
}

TEST(Size_test, alignment) {
  Size v {1, 2, 3, 4};
  Size a {2, 2, 2, 2};
  Size z = align(v, a);
  ASSERT_EQ(z[0], 2);
  ASSERT_EQ(z[1], 2);
  ASSERT_EQ(z[2], 4);
  ASSERT_EQ(z[3], 4);
}

TEST(Size_test, range) {
  Size r0 {-4, -2, 0, 2};
  Size r1 {4, 2, 0, -2};
  Size z = Size::Zeros(4);
  z.range(-4, 2);
  ASSERT_EQ(z, r0);
  z.range(4, -2);
  ASSERT_EQ(z, r1);
}

TEST(Size_test, Range) {
  ASSERT_EQ(Size::Range(-4, 2), Size({-4, -3, -2, -1, 0, 1}));
  ASSERT_EQ(Size::Range(4, -2), Size({}));
  ASSERT_EQ(Size::Range(-4, 2, 2), Size({-4, -2, 0}));
  ASSERT_EQ(Size::Range(4, -2, -2), Size({4, 2, 0}));
  ASSERT_EQ(Size::Range(-4, 2, 5), Size({-4, 1}));
  ASSERT_EQ(Size::Range(-4, 2, 6), Size({-4}));
  ASSERT_EQ(Size::Range(4, -2, -5), Size({4, -1}));
  ASSERT_EQ(Size::Range(4), Size({0, 1, 2, 3}));
  ASSERT_EQ(Size::Range(-4), Size({}));
}