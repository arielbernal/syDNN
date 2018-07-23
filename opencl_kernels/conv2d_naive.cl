// Copyright (c) 2016-2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//#include "include/include_all.cl"

__kernel void convolution(
    __global INPUT_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
    )
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_B == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % OUTPUT_F;
    const uint b = get_global_id(2) / OUTPUT_F;
#endif

    COMPUTE_TYPE acc = (COMPUTE_TYPE)(0);
    for (uint fc = 0; fc < FILTER_F; ++fc) {
        for (uint fy = 0; fy < FILTER_Y ; ++fy) {
            for (uint fx = 0; fx < FILTER_X ; ++fx) {
                int iy = y * STRIDE_Y + INPUT_Y_OFFSET + fy * DILATION_Y;
                int ix = x * STRIDE_X + INPUT_X_OFFSET + fx * DILATION_X;
                uint input_idx = ix * INPUT_X_PITCH + iy * INPUT_Y_PITCH + fc * INPUT_F_PITCH + b * INPUT_B_PITCH;
                uint filter_idx = fx * FILTER_X_PITCH + fy * FILTER_Y_PITCH + fc * FILTER_F_PITCH + f * FILTER_B_PITCH;
                acc += input[input_idx] * weights[filter_idx];
            }
        }
    }

#if BIAS_TERM
    acc += bias[f];
#endif
    uint out_index = b * OUTPUT_B_PITCH + f * OUTPUT_F_PITCH + (y + OUTPUT_Y_PADDING) * OUTPUT_Y_PITCH +
                     (x + OUTPUT_X_PADDING) * OUTPUT_X_PITCH;
    output[out_index] = (OUTPUT_TYPE)(acc);
    //ACTIVATION(dotProd, NL_M, NL_N);
}
