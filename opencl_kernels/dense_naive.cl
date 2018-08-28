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

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define GET_DATA_INDEX_2D(prefix, y, x)  \
    (x)*CAT(prefix, _X_PITCH) +       \
    (y)*CAT(prefix, _Y_PITCH)

__kernel void dense(
    __global const INPUT_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global const FILTER_TYPE* weights
#if BIAS_TERM
    , __global const BIAS_TYPE* bias
#endif
    )
{
    const uint y = get_global_id(0);
#if OUTPUT_Y == 1
    const uint b = 0;
#else
    const uint b = get_global_id(1);
#endif
    COMPUTE_TYPE acc = (COMPUTE_TYPE)(0);
    for (uint x = 0; x < INPUT_X; ++x) {
        uint input_idx = GET_DATA_INDEX_2D(INPUT, b, x) ;
        uint filter_idx = GET_DATA_INDEX_2D(FILTER, x, y);
        acc += input[input_idx] * weights[filter_idx];
    }

#if BIAS_TERM
    acc += bias[y];
#endif
    uint out_index = GET_DATA_INDEX_2D(OUTPUT, b, y);
    output[out_index] = (OUTPUT_TYPE)(acc);
    //ACTIVATION(dotProd, NL_M, NL_N);
}
