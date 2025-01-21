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

#pragma once

#include <limits>
#include <NvInferVersion.h>

#include "proto/rt.pb.h"

#include "common/inference/tensorrt/rt_common.h"

namespace apollo {
namespace perception {
namespace inference {

void cmp(int block_size, int thread_size, int shared_memory_size,
                cudaStream_t &stream, const int nthreads, const float *in_data,
                const int channels, const int height, const int width,
                const bool out_max_val, float *out_data, const float float_min);

}  // namespace inference
}  // namespace perception
}  // namespace apollo
