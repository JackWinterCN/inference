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

#pragma once

#include <algorithm>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/rt_common.h"
#include "common/inference/tensorrt/plugins/kernels.h"

namespace apollo {
namespace perception {
namespace inference {

void generate_anchors(int block_size, int thread_size, int shared_memory_size,
                      cudaStream_t &stream, const int height, const int width,
                      const float anchor_stride, const int num_anchor_per_point,
                      const float *anchor_heights, const float *anchor_widths,
                      float *anchors);

// template <typename Dtype>
// void reshape_boxes(int block_size, int thread_size, int shared_memory_size,
//                    cudaStream_t &stream, const int nthreads,
//                    const Dtype *in_boxes, const int height, const int width,
//                    const int num_box_per_point, Dtype *out_boxes);

void reshape_boxes(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads,
                   const float *in_boxes, const int height, const int width,
                   const int num_box_per_point, float *out_boxes);

// template <typename Dtype>
// void reshape_scores(int block_size, int thread_size, int shared_memory_size,
//                     cudaStream_t &stream, const int nthreads,
//                     const Dtype *in_scores, const int height, const int width,
//                     const int num_box_per_point, Dtype *out_scores);

void reshape_scores(int block_size, int thread_size, int shared_memory_size,
                    cudaStream_t &stream, const int nthreads,
                    const float *in_scores, const int height, const int width,
                    const int num_box_per_point, float *out_scores);

}  // namespace inference
}  // namespace perception
}  // namespace apollo
