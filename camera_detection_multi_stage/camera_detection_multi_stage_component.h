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
#pragma once

#include <memory>
#include <string>

#include "proto/sensor_image.pb.h"
#include "proto/camera_detection_multi_stage.pb.h"

#include "cyber/cyber.h"
#include "common/interface/base_obstacle_detector.h"
#include "common/onboard/inner_component_messages/camera_detection_component_messages.h"
// #include "common/onboard/transform_wrapper/transform_wrapper.h"

namespace apollo {
namespace perception {
namespace camera {

class CameraDetectionMultiStageComponent final
    : public cyber::Component<drivers::Image> {
 public:
  CameraDetectionMultiStageComponent() = default;
  ~CameraDetectionMultiStageComponent() = default;

  /**
   * @brief set config file path
   *
   * @return true
   * @return false
   */
  void SetConfigFilePath(const std::string& conf_path);

  /**
   * @brief Init for amera detection 2d compoment
   *
   * @return true
   * @return false
   */
  bool Init() override;
  /**
   * @brief Process of camera detection 2d compoment
   *
   * @param msg image msg
   * @return true
   * @return false
   */
  bool Proc(const std::shared_ptr<drivers::Image>& msg) override;
  /**
   * @brief run local test
   *
   * @return true
   * @return false
   */
  bool RunTest();

 private:
  bool InitCameraFrame();

  bool InitObstacleDetector();

  // bool InitTransformWrapper(const CameraDetectionMultiStage& detection_param);

  bool InternalProc(const std::shared_ptr<apollo::drivers::Image>& msg,
                    const std::shared_ptr<onboard::CameraFrame>& out_message);

 private:
  std::string conf_path_{"camera_detection_multi_stage_config.pb.txt"};
  CameraDetectionMultiStage detection_param_;
  int image_height_;
  int image_width_;
  int frame_id_ = 0;
  double timestamp_offset_;
  Eigen::Matrix3f camera_k_matrix_ = Eigen::Matrix3f::Identity();

  std::shared_ptr<BaseObstacleDetector> detector_;
  std::shared_ptr<camera::DataProvider> data_provider_;

  std::shared_ptr<cyber::Writer<onboard::CameraFrame>> writer_;
  // std::shared_ptr<onboard::TransformWrapper> trans_wrapper_;

  DISALLOW_COPY_AND_ASSIGN(CameraDetectionMultiStageComponent);
};

CYBER_REGISTER_COMPONENT(CameraDetectionMultiStageComponent);

}  // namespace camera
}  // namespace perception
}  // namespace apollo
