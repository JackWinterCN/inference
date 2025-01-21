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

/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <NvInferVersion.h>

#include "common/inference/tensorrt/plugins/dfmb_psroi_align_plugin.h"
#include "common/inference/tensorrt/plugins/dfmb_psroi_align_plugin_kernel.h"
#include "common/inference/tensorrt/plugins/kernels.h"

namespace apollo {
namespace perception {
namespace inference {


#ifdef NV_TENSORRT_MAJOR
#if NV_TENSORRT_MAJOR != 8
int DFMBPSROIAlignPlugin::enqueue(int batchSize, const void *const *inputs,
                                  void **outputs, void *workspace,
                                  cudaStream_t stream) {
#else
int32_t DFMBPSROIAlignPlugin::enqueue(int32_t batchSize, const void *const *inputs, void *const *outputs,
                      void *workspace, cudaStream_t stream) noexcept {
#endif
#endif
  const float *bottom_data = reinterpret_cast<const float *>(inputs[0]);
  const float *bottom_rois = reinterpret_cast<const float *>(inputs[1]);
  const float *bottom_trans =
      no_trans_ ? nullptr : reinterpret_cast<const float *>(inputs[2]);
  float *top_data = reinterpret_cast<float *>(outputs[0]);
  int channels_each_class =
      no_trans_ ? output_channel_ : output_channel_ / num_classes_;

  BASE_GPU_CHECK(
      cudaMemsetAsync(top_data, 0, output_size_ * sizeof(float), stream));
  BASE_GPU_CHECK(cudaDeviceSynchronize());

  int block_size = (output_size_ - 1) / thread_size_ + 1;
  DFMBPSROIAlignForward(block_size, thread_size_, 0, stream, output_size_,
                        bottom_data, heat_map_a_, heat_map_b_, pad_ratio_,
                        batchSize, channels_, height_, width_, pooled_height_,
                        pooled_width_, bottom_rois, bottom_trans, no_trans_,
                        trans_std_, sample_per_part_, output_channel_,
                        group_height_, group_width_, part_height_, part_width_,
                        num_classes_, channels_each_class, top_data);

  return 0;
}
}  // namespace inference
}  // namespace perception
}  // namespace apollo
