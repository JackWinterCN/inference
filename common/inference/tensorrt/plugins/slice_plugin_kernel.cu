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

#include <NvInferVersion.h>

#include <vector>

#include "common/inference/tensorrt/plugins/slice_plugin_kernel.h"

namespace apollo {
namespace perception {
namespace inference {

typedef int8_t int8;

template <typename Dtype>
__global__ void Slice_kernel(const int nthreads, const Dtype *in_data,
                      const int num_slices, const int slice_size,
                      const int bottom_slice_axis, const int top_slice_axis,
                      const int offset_slice_axis, Dtype *out_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index =
        slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    out_data[index] = in_data[bottom_index];
  }
}

// template <typename Dtype>
// void Slice(int block_size, int thread_size, int shared_memory_size,
//            cudaStream_t &stream, const int nthreads, const Dtype *in_data,
//            const int num_slices, const int slice_size,
//            const int bottom_slice_axis, const int top_slice_axis,
//            const int offset_slice_axis, Dtype *out_data) {
//   Slice_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
//       nthreads, in_data, num_slices, slice_size, bottom_slice_axis,
//       top_slice_axis, offset_slice_axis, out_data);
// }

void Slice(int block_size, int thread_size, int shared_memory_size,
           cudaStream_t &stream, const int nthreads, const float *in_data,
           const int num_slices, const int slice_size,
           const int bottom_slice_axis, const int top_slice_axis,
           const int offset_slice_axis, float *out_data) {
  Slice_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, in_data, num_slices, slice_size, bottom_slice_axis,
      top_slice_axis, offset_slice_axis, out_data);
}

}  // namespace inference
}  // namespace perception
}  // namespace apollo
