/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
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
#include "common/inference/tensorrt/plugins/rpn_proposal_ssd_plugin_kernel.h"

#include <thrust/sort.h>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/plugins/kernels.h"

namespace apollo {
namespace perception {
namespace inference {

// TODO(chenjiahao): add heat_map_b as anchor_offset
// output anchors dims: [H, W, num_anchor_per_point, 4]
__global__ void generate_anchors_kernel(const int height, const int width,
                                        const float anchor_stride,
                                        const int num_anchor_per_point,
                                        const float *anchor_heights,
                                        const float *anchor_widths,
                                        float *anchors) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_anchor = height * width * num_anchor_per_point;
  if (index >= num_anchor) {
    return;
  }

  float anchor_offset = 0;
  int pos_index = index / num_anchor_per_point;
  int anchor_id = index % num_anchor_per_point;
  int w_i = pos_index % width;
  int h_i = pos_index / width;

  // center coordinates
  float x_ctr = w_i * anchor_stride + anchor_offset;
  float y_ctr = h_i * anchor_stride + anchor_offset;

  float x_min = x_ctr - 0.5 * (anchor_widths[anchor_id] - 1);
  float y_min = y_ctr - 0.5 * (anchor_heights[anchor_id] - 1);
  float x_max = x_ctr + 0.5 * (anchor_widths[anchor_id] - 1);
  float y_max = y_ctr + 0.5 * (anchor_heights[anchor_id] - 1);

  anchors[index * 4] = x_min;
  anchors[index * 4 + 1] = y_min;
  anchors[index * 4 + 2] = x_max;
  anchors[index * 4 + 3] = y_max;
}

void generate_anchors(int block_size, int thread_size, int shared_memory_size,
                      cudaStream_t &stream, const int height, const int width,
                      const float anchor_stride, const int num_anchor_per_point,
                      const float *anchor_heights, const float *anchor_widths,
                      float *anchors) {
  generate_anchors_kernel<<<block_size, thread_size, shared_memory_size,
                            stream>>>(height, width, anchor_stride,
                                      num_anchor_per_point, anchor_heights,
                                      anchor_widths, anchors);
}
// in_boxes dims: [N, num_box_per_point * 4, H, W],
// out_boxes dims: [N, H * W * num_box_per_point， 4]
template <typename Dtype>
__global__ void reshape_boxes_kernel(const int nthreads, const Dtype *in_boxes,
                                     const int height, const int width,
                                     const int num_box_per_point,
                                     Dtype *out_boxes) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int num_point = height * width;

    int batch_id = index / num_point / num_box_per_point / 4;
    int feature_id = index % 4;
    int box_id = (index / 4) % num_box_per_point;
    int point_id = (index / num_box_per_point / 4) % num_point;

    int in_index =
        ((batch_id * num_box_per_point + box_id) * 4 + feature_id) * num_point +
        point_id;
    out_boxes[index] = in_boxes[in_index];
  }
}
// template <typename Dtype>
// void reshape_boxes(int block_size, int thread_size, int shared_memory_size,
//                    cudaStream_t &stream, const int nthreads,
//                    const Dtype *in_boxes, const int height, const int width,
//                    const int num_box_per_point, Dtype *out_boxes) {
//   reshape_boxes_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
//       nthreads, in_boxes, height, width, num_box_per_point, out_boxes);
// }

void reshape_boxes(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads,
                   const float *in_boxes, const int height, const int width,
                   const int num_box_per_point, float *out_boxes) {
  reshape_boxes_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, in_boxes, height, width, num_box_per_point, out_boxes);
}

// in_scores dims: [N, 2 * num_box_per_point, H, W],
// out_scores dims: [N, H * W * num_box_per_point, 2]
template <typename Dtype>
__global__ void reshape_scores_kernel(const int nthreads,
                                      const Dtype *in_scores, const int height,
                                      const int width,
                                      const int num_box_per_point,
                                      Dtype *out_scores) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int num_point = height * width;

    int batch_id = index / num_point / num_box_per_point / 2;
    int class_id = index % 2;
    int box_id = (index / 2) % num_box_per_point;
    int point_id = (index / num_box_per_point / 2) % num_point;

    int in_index =
        ((batch_id * 2 + class_id) * num_box_per_point + box_id) * num_point +
        point_id;
    out_scores[index] = in_scores[in_index];
  }
}

// template <typename Dtype>
// void reshape_scores(int block_size, int thread_size, int shared_memory_size,
//                     cudaStream_t &stream, const int nthreads,
//                     const Dtype *in_scores, const int height, const int width,
//                     const int num_box_per_point, Dtype *out_scores) {
//   reshape_scores_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
//       nthreads, in_scores, height, width, num_box_per_point,
//       out_scores);
// }

void reshape_scores(int block_size, int thread_size, int shared_memory_size,
                    cudaStream_t &stream, const int nthreads,
                    const float *in_scores, const int height, const int width,
                    const int num_box_per_point, float *out_scores) {
  reshape_scores_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, in_scores, height, width, num_box_per_point,
      out_scores);
}
}  // namespace inference
}  // namespace perception
}  // namespace apollo
