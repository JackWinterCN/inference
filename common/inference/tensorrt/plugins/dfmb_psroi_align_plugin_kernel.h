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

#include <NvInferVersion.h>

#include "cyber/common/log.h"
#include "common/inference/tensorrt/rt_common.h"

namespace apollo {
namespace perception {
namespace inference {
// template <typename Dtype>
// void DFMBPSROIAlignForward(
//     int block_size, int thread_size, int shared_memory_size,
//     cudaStream_t &stream, const int nthreads, const Dtype *bottom_data,
//     const Dtype heat_map_a, const Dtype heat_map_b, const Dtype pad_ratio,
//     const int batch_size, const int channels, const int height, const int width,
//     const int pooled_height, const int pooled_width, const Dtype *bottom_rois,
//     const Dtype *bottom_trans, const bool no_trans, const Dtype trans_std,
//     const int sample_per_part, const int output_channel, const int group_height,
//     const int group_width, const int part_height, const int part_width,
//     const int num_classes, const int channels_each_class, Dtype *top_data);

void DFMBPSROIAlignForward(
    int block_size, int thread_size, int shared_memory_size,
    cudaStream_t &stream, const int nthreads, const float *bottom_data,
    const float heat_map_a, const float heat_map_b, const float pad_ratio,
    const int batch_size, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const float *bottom_rois,
    const float *bottom_trans, const bool no_trans, const float trans_std,
    const int sample_per_part, const int output_channel, const int group_height,
    const int group_width, const int part_height, const int part_width,
    const int num_classes, const int channels_each_class, float *top_data);

}  // namespace inference
}  // namespace perception
}  // namespace apollo
