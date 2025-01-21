/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
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

#include <vector>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/plugins/leakyReLU_plugin_kernel.h"

namespace apollo {
namespace perception {
namespace inference {

template <typename Dtype>
__global__ void ReLU_kernel(const int nthreads, const Dtype *in_data,
                     const float negative_slope, Dtype *out_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    out_data[index] = in_data[index];
    if (out_data[index] < 0.0) {
      out_data[index] *= negative_slope;
    }
  }
}

// template <typename Dtype>
// void ReLU(int block_size, int thread_size, int shared_memory_size,
//           cudaStream_t &stream, const int nthreads, const Dtype *in_data,
//           const float negative_slope, Dtype *out_data) {
//   ReLU<<<block_size, thread_size, shared_memory_size, stream>>>(
//       nthreads, in_data, negative_slope, out_data);
// }

void ReLU(int block_size, int thread_size, int shared_memory_size,
          cudaStream_t &stream, const int nthreads, const float *in_data,
          const float negative_slope, float *out_data) {
  ReLU_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, in_data, negative_slope, out_data);
}

}  // namespace inference
}  // namespace perception
}  // namespace apollo
