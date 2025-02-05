/******************************************************************************
 * Copyright 2023 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include <memory>
#include <string>

#include "camera_detection_multi_stage/camera_detection_multi_stage_component.h"

using apollo::perception::camera::CameraDetectionMultiStageComponent;

int main(int argc, char** argv) {
  apollo::cyber::binary::SetName("InferenceDemo");
  std::string conf_path =
    "./camera_detection_multi_stage/conf/camera_detection_multi_stage_config.pb.txt";
  if (argc > 1) {
    conf_path = std::string(argv[1]);
  }
  CameraDetectionMultiStageComponent infer_test;
  infer_test.SetConfigFilePath(conf_path);
  if (!infer_test.Init()) {
    AERROR << "CameraDetectionMultiStageComponent init failed!";
    return 1;
  }
  infer_test.RunTest();
  return 0;
}