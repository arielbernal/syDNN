#include <common/tensor_ref.hpp>
#include <common/test_common.hpp>
#include <size.hpp>


using namespace syDNN;
using namespace syDNN::test;

template<typename T>
TensorRef<T> conv2d_ref(TensorRef<T>& input, TensorRef<T>& weights, TensorRef<T>& bias,
                      size_t output_padding_y = 0, size_t output_padding_x = 0,
                      int stride_x = 1, int stride_y = 1, int dilation_y = 1, int dilation_x = 1)
{
  size_t input_b = input.shape(0);
  size_t input_c = input.shape(1);
  size_t input_y = input.shape(2);
  size_t input_x = input.shape(3);
  size_t input_padding_y = input.padding(2);
  size_t input_padding_x = input.padding(3);

  size_t weights_f = weights.shape(0);
  size_t weights_c = weights.shape(1);
  size_t weights_y = weights.shape(2);
  size_t weights_x = weights.shape(3);

  size_t output_b = input.shape(0);
  size_t output_c = weights_f;
  size_t output_y = 1 + (int(input_y) + 2 * input_padding_y - ((weights_y - 1) * dilation_y + 1)) / stride_y;
  size_t output_x = 1 + (int(input_x) + 2 * input_padding_x - ((weights_x - 1) * dilation_x + 1)) / stride_x;

  TensorRef<T> output({int32_t(output_b), int32_t(output_c), int32_t(output_y), int32_t(output_x)},
                      {0, 0, int32_t(output_padding_y), int32_t(output_padding_x)});

  for (size_t b = 0; b < output_b; ++b) {
    for (size_t c = 0; c < output_c; ++c) {
      for (size_t y = 0; y < output_y; ++y) {
        for (size_t x = 0; x < output_x; ++x) {
          T acc = 0;
          for (size_t fc = 0; fc < weights_c; ++fc) {
            for (size_t fy = 0; fy < weights_y; ++fy) {
              for (size_t fx = 0; fx < weights_x; ++fx) {
                int iy = int(stride_y) * y + fy * dilation_y + input_padding_y;
                int ix = int(stride_x) * x + fx * dilation_x + input_padding_x;
                if (ix < 0 || iy < 0) continue;
                acc += input(b, fc, iy, ix) * weights(c, fc, fy, fx);
              } // fx
            } // fy
          } // fc
          output(b, c, y, x) = acc + bias(c);
        } // x
      } // y
    } // c
  } // b
  return output;
}



TEST(conv2d_test, first) {

  TensorRefF input = RandomTensorRef({1, 2, 5, 5}, {0, 0, 0, 0}, 1, 10);
  std::cout << input.to_string("%4.0f") << std::endl;
  TensorRefF filter = RandomTensorRef({3, 2, 3, 3}, {0, 0, 0, 0}, 1, 10);
  std::cout << filter.to_string("%4.0f") << std::endl;
  TensorRefF bias = RandomTensorRef({3}, {0}, 1, 10);
  std::cout << bias.to_string("%4.0f") << std::endl;

  auto out = conv2d_ref(input, filter, bias, 0, 0, 1, 1, 1, 1);
  std::cout << out.to_string("%4.0f") << std::endl;

  auto out1 = conv2d_ref(input, filter, bias, 0, 0, 1, 1, 2, 2);
  std::cout << out1.to_string("%4.0f") << std::endl;

}