/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include <limits>
#include <vector>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/plugins/argmax_plugin_kernel.h"
namespace apollo {
namespace perception {
namespace inference {
__global__ void cmp_kernel(const int nthreads, const float *in_data,
                    const int channels, const int height, const int width,
                    const bool out_max_val, float *out_data,
                    const float float_min) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nthreads) {
    int w = idx % width;
    idx = idx / width;
    int h = idx % height;
    idx = idx / height;
    int c = idx % channels;
    int n = idx / channels;
    if (c != 0) {
      return;
    }
    int c_max = 0;
    float v_max = float_min;
    for (int c = 0; c < channels; ++c) {
      int in_idx = ((n * channels + c) * height + h) * width + w;
      if (v_max < in_data[in_idx]) {
        v_max = in_data[in_idx];
        c_max = c;
      }
    }
    int out_idx_idx = ((n * channels + 0) * height + h) * width + w;
    out_data[out_idx_idx] = c_max;
    if (out_max_val) {
      int out_val_idx = ((n * channels + 1) * height + h) * width + w;
      out_data[out_val_idx] = v_max;
    }
  }
}

void cmp(int block_size, int thread_size, int shared_memory_size,
                cudaStream_t &stream, const int nthreads, const float *in_data,
                const int channels, const int height, const int width,
                const bool out_max_val, float *out_data,
                const float float_min) {
  cmp_kernel<<<block_size, thread_size>>>(nthreads, in_data, channels, height, width,
                                   out_max_val, out_data, float_min);
}

}  // namespace inference
}  // namespace perception
}  // namespace apollo
