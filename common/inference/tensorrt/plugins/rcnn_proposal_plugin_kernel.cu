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

#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <NvInferVersion.h>

#include "common/inference/tensorrt/plugins/kernels.h"
#include "common/inference/tensorrt/plugins/rcnn_proposal_plugin_kernel.h"

namespace apollo {
namespace perception {
namespace inference {

// nthreads = num_rois
__global__ void get_rois_nums_kernel(const int nthreads, const float *rois,
                                     int *batch_rois_nums) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int batch_id = (int)rois[index * 5];
    if (batch_id >= 0) {
      atomicAdd(&batch_rois_nums[batch_id], 1);
    }
  }
}

void get_rois_nums(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads, const float *rois,
                   int *batch_rois_nums) {
  get_rois_nums_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, rois, batch_rois_nums);
}

// bbox_pred dims: [num_rois, box_len, num_class]
// out_bbox_pred dims: [num_rois, num_class, box_len]
__global__ void transpose_bbox_pred_kernel(const int nthreads,
                                           const float *bbox_pred,
                                           const int box_len,
                                           const int num_class,
                                           float *out_bbox_pred) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int roi_id = index / num_class / box_len;
    int class_id = (index / box_len) % num_class;
    int feature_id = index % box_len;

    int in_index =
        roi_id * box_len * num_class + feature_id * num_class + class_id;
    out_bbox_pred[index] = bbox_pred[in_index];
  }
}

void transpose_bbox_pred(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads,
                                           const float *bbox_pred,
                                           const int box_len,
                                           const int num_class,
                                           float *out_bbox_pred) {
 transpose_bbox_pred_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, bbox_pred, box_len, num_class, out_bbox_pred);
}

// bbox_pred dims: [num_box, num_class+1, 4],
// scores dims: [num_box, num_class+1],
// out_bbox_pred dims: [num_box, 4]
// out_scores dims: [num_box]
__global__ void get_max_score_kernel(const int nthreads, const float *bbox_pred,
                                     const float *scores, const int num_class,
                                     const float threshold_objectness,
                                     const float *class_thresholds,
                                     float *out_bbox_pred, float *out_scores,
                                     float *out_all_probs, int *filter_count) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= nthreads) {
    return;
  }

  int box_id = index;
  if ((1.0f - scores[box_id * (num_class + 1)]) < threshold_objectness) {
    return;
  }

  float score_max = -FLT_MAX;
  int cls_max = -1;
  for (int c = 0; c < num_class; ++c) {
    float score =
        scores[box_id * (num_class + 1) + c + 1] - class_thresholds[c];
    if (score > score_max) {
      score_max = score;
      cls_max = c;
    }
  }
  if (score_max < 0) {
    return;
  } else {
    int counter = atomicAdd(filter_count, 1);
    int box_cls_id = box_id * (num_class + 1) + cls_max + 1;
    for (int i = 0; i < 4; ++i) {
      out_bbox_pred[counter * 4 + i] = bbox_pred[box_cls_id * 4 + i];
    }
    out_scores[counter] = scores[box_cls_id];
    for (int i = 0; i < num_class + 1; ++i) {
      out_all_probs[counter * (num_class + 1) + i] =
          scores[box_id * (num_class + 1) + i];
    }
  }
}

void get_max_score(int block_size, int thread_size, int shared_memory_size,
                   cudaStream_t &stream, const int nthreads, const float *bbox_pred,
                                     const float *scores, const int num_class,
                                     const float threshold_objectness,
                                     const float *class_thresholds,
                                     float *out_bbox_pred, float *out_scores,
                                     float *out_all_probs, int *filter_count) {
  get_max_score_kernel<<<block_size, thread_size, shared_memory_size, stream>>>(
      nthreads, bbox_pred, scores, num_class, threshold_objectness,
      class_thresholds, out_bbox_pred, out_scores, out_all_probs, filter_count);
}

}  // namespace inference
}  // namespace perception
}  // namespace apollo
