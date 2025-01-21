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

#include <vector>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/rt_common.h"

namespace apollo {
namespace perception {
namespace inference {


void get_rois_nums(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads, const float *rois,
                   int *batch_rois_nums);

void transpose_bbox_pred(int block_size, int thread_size,
                         int shared_memory_size, cudaStream_t &stream,
                         const int nthreads, const float *bbox_pred,
                         const int box_len, const int num_class,
                         float *out_bbox_pred);

void get_max_score(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads,
                   const float *bbox_pred, const float *scores,
                   const int num_class, const float threshold_objectness,
                   const float *class_thresholds, float *out_bbox_pred,
                   float *out_scores, float *out_all_probs, int *filter_count);
}  // namespace inference
}  // namespace perception
}  // namespace apollo
