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

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define GET_DATA_INDEX(prefix, b, f, y, x)  \
    (x)*CAT(prefix, _X_PITCH) +             \
    (y)*CAT(prefix, _Y_PITCH) +             \
    (f)*CAT(prefix, _F_PITCH) +             \
    (b)*CAT(prefix, _B_PITCH)

__kernel void convolution(
    __global const INPUT_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global const FILTER_TYPE* weights
#if BIAS_TERM
    , __global const BIAS_TYPE* bias
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
                uint input_idx = GET_DATA_INDEX(INPUT, b, fc, iy, ix);
                uint filter_idx = GET_DATA_INDEX(FILTER, f, fc, fy, fx);
                acc += input[input_idx] * weights[filter_idx];
            }
        }
    }

#if BIAS_TERM
    acc += bias[f];
#endif
    uint out_index = GET_DATA_INDEX(OUTPUT, b, f, (y + OUTPUT_Y_PADDING), (x + OUTPUT_X_PADDING));
    output[out_index] = (OUTPUT_TYPE)(acc);
    //ACTIVATION(dotProd, NL_M, NL_N);
}
