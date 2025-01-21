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
#include "camera_detection_multi_stage/camera_detection_multi_stage_component.h"

#include "cyber/profiler/profiler.h"
#include "common/algorithm/sensor_manager/sensor_manager.h"
#include "common/base/camera.h"
#include "common/camera/common/data_provider.h"
#include "detector/yolo/yolo_obstacle_detector.h"
#include "cyber/cyber.h"
#include "opencv2/opencv.hpp"
namespace apollo {
namespace perception {
namespace camera {

bool CameraDetectionMultiStageComponent::InitObstacleDetector(
    const CameraDetectionMultiStage& detection_param) {
  ObstacleDetectorInitOptions init_options;
  // Init conf file
  auto plugin_param = detection_param.plugin_param();
  init_options.config_path = plugin_param.config_path();
  init_options.config_file = plugin_param.config_file();
  init_options.gpu_id = detection_param.gpu_id();
  timestamp_offset_ = detection_param.timestamp_offset();

  // Init camera params
  std::string camera_name = detection_param.camera_name();
  base::BaseCameraModelPtr model =
      algorithm::SensorManager::Instance()->GetUndistortCameraModel(
          camera_name);
  ACHECK(model) << "Can't find " << camera_name
                << " in data/conf/sensor_meta.pb.txt";
  auto pinhole = static_cast<base::PinholeCameraModel*>(model.get());
  init_options.intrinsic = pinhole->get_intrinsic_params();
  camera_k_matrix_ = init_options.intrinsic;
  init_options.image_height = model->get_height();
  init_options.image_width = model->get_width();
  image_height_ = model->get_height();
  image_width_ = model->get_width();
  AERROR << "image_height_: " << image_height_ << " image_width_: "
        << image_width_;
  // Init detector
  // RegisterFactoryYoloObstacleDetector1 temp;
  // {
  //   ::apollo::perception::lib::FactoryMap &map =
  //       ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"];
  //   std::cout << "============> register class: " << "YoloObstacleDetector"
  //             << std::endl;
  //   if (map.find("YoloObstacleDetector") == map.end()) {
  //     map["YoloObstacleDetector"] = new ObjectFactoryYoloObstacleDetector();
  //   }
  // }
  // detector_.reset(
  //     BaseObstacleDetectorRegisterer::GetInstanceByName(plugin_param.name()));
  detector_.reset(new YoloObstacleDetector());
  detector_->Init(init_options);
  return true;
}

bool CameraDetectionMultiStageComponent::InitCameraFrame(
    const CameraDetectionMultiStage& detection_param) {
  DataProvider::InitOptions init_options;
  init_options.image_height = image_height_;
  init_options.image_width = image_width_;
  init_options.do_undistortion = detection_param.enable_undistortion();
  init_options.sensor_name = detection_param.camera_name();
  init_options.device_id = detection_param.gpu_id();
  AINFO << "init_options.device_id: " << init_options.device_id
        << " camera_name: " << init_options.sensor_name;

  data_provider_ = std::make_shared<camera::DataProvider>();
  data_provider_->Init(init_options);

  return true;
}

// bool CameraDetectionMultiStageComponent::InitTransformWrapper(
//     const CameraDetectionMultiStage& detection_param) {
//   trans_wrapper_.reset(new onboard::TransformWrapper());
//   // tf_camera_frame_id
//   trans_wrapper_->Init(detection_param.camera_name());
//   return true;
// }

bool CameraDetectionMultiStageComponent::Init() {
  google::SetStderrLogging(0);
  CameraDetectionMultiStage detection_param;

  if (!cyber::common::GetProtoFromFile(
          "/apollo_workspace/inference/camera_detection_multi_stage/conf/camera_detection_multi_stage_config.pb.txt",
          &detection_param)) {
    AERROR << "Load camera detection 3d component config failed!";
    return false;
  }

  InitObstacleDetector(detection_param);

  InitCameraFrame(detection_param);

  // InitTransformWrapper(detection_param);

  // writer_ = node_->CreateWriter<onboard::CameraFrame>(
  //     detection_param.channel().output_obstacles_channel_name());

  test();
  return true;
}

bool CameraDetectionMultiStageComponent::Proc(
    const std::shared_ptr<apollo::drivers::Image>& msg) {
  PERF_FUNCTION()
  std::shared_ptr<onboard::CameraFrame> out_message(new (std::nothrow)
                                                        onboard::CameraFrame);
  bool status = InternalProc(msg, out_message);
  if (status) {
    writer_->Write(out_message);
    AINFO << "Send camera detection 2d output message.";
  }

  return status;
}

bool CameraDetectionMultiStageComponent::InternalProc(
    const std::shared_ptr<apollo::drivers::Image>& msg,
    const std::shared_ptr<onboard::CameraFrame>& out_message) {
  out_message->data_provider = data_provider_;
  // Fill image
  // todo(daohu527): need use real memory size
  out_message->data_provider->FillImageData(
      image_height_, image_width_,
      reinterpret_cast<const uint8_t*>(msg->data().data()), msg->encoding());

  out_message->camera_k_matrix = camera_k_matrix_;

  const double msg_timestamp = msg->measurement_time() + timestamp_offset_;
  out_message->timestamp = msg_timestamp;

  // Get sensor to world pose from TF
  Eigen::Affine3d camera2world;
  // if (!trans_wrapper_->GetSensor2worldTrans(msg_timestamp, &camera2world)) {
  //   const std::string err_str =
  //       absl::StrCat("failed to get camera to world pose, ts: ", msg_timestamp,
  //                    " frame_id: ", msg->frame_id());
  //   AERROR << err_str;
  //   return false;
  // }

  out_message->frame_id = frame_id_;
  ++frame_id_;

  out_message->camera2world_pose = camera2world;

  // Detect
  PERF_BLOCK("camera_2d_detector")
  detector_->Detect(out_message.get());
  PERF_BLOCK_END
  return true;
}


int CameraDetectionMultiStageComponent::test() {
  AINFO << "============================================> start test";
  AINFO << "============================================> start test";
  std::shared_ptr<onboard::CameraFrame> out_message(new (std::nothrow)
                                                        onboard::CameraFrame);
  const int height = 576;
  const int width = 1440;
  const int offset_y = 312;
  std::vector<int> shape = {1, height, width, 3};
  std::map<std::string, std::vector<int>> shape_map{{"data", shape}};
  
  std::vector<std::string> image_lists;

  const int count = 3 * width * height;
  std::vector<float> output_data_vec;
  cv::Mat img = cv::imread("/apollo_workspace/modules/perception/common/inference/inference_test_data/images/ARZ034_12_1499218335_1499218635_500.jpg");
  AINFO << "init img.cols: " << img.cols << " img.rows: " << img.rows;
  // cv::Rect roi(0, offset_y, img.cols, img.rows - offset_y);
  // cv::Mat img_roi = img(roi);
  // img_roi.copyTo(img);
  // cv::resize(img, img, cv::Size(width, height));
  // AINFO << "resized img.cols: " << img.cols << " img.rows: " << img.rows;
  // cv::imwrite("/apollo_workspace/modules/perception/common/inference/inference_test_data/images/ARZ034_12_1499218335_1499218635_500_resize.jpg", img);
  // AINFO << "have saved resized image";
  out_message->data_provider = data_provider_;
  // Fill image
  // todo(daohu527): need use real memory size
  out_message->data_provider->FillImageData(
      image_height_, image_width_, reinterpret_cast<const uint8_t*>(img.data),
      "bgr8");
  out_message->camera_k_matrix = camera_k_matrix_;

  // Get sensor to world pose from TF
  // Eigen::Affine3d camera2world;
  // if (!trans_wrapper_->GetSensor2worldTrans(msg_timestamp, &camera2world)) {
  //   const std::string err_str =
  //       absl::StrCat("failed to get camera to world pose");
  //   AERROR << err_str;
  //   return false;
  // }
  // out_message->camera2world_pose = camera2world;

  out_message->frame_id = frame_id_;
  ++frame_id_;


  // Detect
  PERF_BLOCK("camera_2d_detector")
  detector_->Detect(out_message.get());
  PERF_BLOCK_END

  for (auto& obj : out_message->detected_objects) {
    AINFO << "obj[" << static_cast<int>(obj->type) << "]: " << obj->camera_supplement.box.xmin << ", "
          << obj->camera_supplement.box.xmax << ", "
          << obj->camera_supplement.box.ymin << ", "
          << obj->camera_supplement.box.ymax;
    cv::rectangle(
        img,
        cv::Point(static_cast<int>(obj->camera_supplement.box.xmin), static_cast<int>(obj->camera_supplement.box.ymin)),
        cv::Point(static_cast<int>(obj->camera_supplement.box.xmax), static_cast<int>(obj->camera_supplement.box.ymax)),
        cv::Scalar(0, 0, 0), 8);
  }
  cv::imwrite("/apollo_workspace/inference/inference_test_data/images/ARZ034_12_1499218335_1499218635_500_rectangle.jpg", img);
  AINFO << "have saved image with box";
  return 0;
}


}  // namespace camera
}  // namespace perception
}  // namespace apollo
