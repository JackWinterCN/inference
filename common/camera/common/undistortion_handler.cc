/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
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
#include "common/camera/common/undistortion_handler.h"

#include <vector>

#include "Eigen/Dense"

#include "cyber/common/log.h"
#include "common/algorithm/sensor_manager/sensor_manager.h"
#include "common/camera/common/image_data_operations.h"

namespace apollo {
namespace perception {
namespace camera {

bool UndistortionHandler::set_device(int device) {
  device_ = device;
  auto code = cudaSetDevice(device_);
  if (code != cudaSuccess) {
    AERROR << "cudaSetDevice failed: " << cudaGetErrorString(code);
    return false;
  }
  return true;
}
/* Initialization of the GPU routines for camera data preprocessing
 *
 * Return: 0 - success; other - failure
 * Params: device - dev number of the GPU device
 * Note: returns OK if already been inited.
 */
bool UndistortionHandler::Init(const std::string &sensor_name, int device) {
  if (inited_) {
    return true;
  }

  std::vector<double> D;
  std::vector<double> K;

  algorithm::SensorManager *sensor_manager =
      algorithm::SensorManager::Instance();
  if (!sensor_manager->IsSensorExist(sensor_name)) {
    AERROR << "Sensor '" << sensor_name << "' not exists!";
    return false;
  }

  if (!set_device(device)) {
    return false;
  }

  base::BrownCameraDistortionModelPtr distort_model =
      std::dynamic_pointer_cast<base::BrownCameraDistortionModel>(
          sensor_manager->GetDistortCameraModel(sensor_name));

  height_ = static_cast<int>(distort_model->get_height());
  width_ = static_cast<int>(distort_model->get_width());
  d_mapx_.Reshape({height_, width_});
  d_mapy_.Reshape({height_, width_});

  Eigen::Matrix3f I;
  I << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f;

  InitUndistortRectifyMap(distort_model->get_intrinsic_params(),
                          distort_model->get_distort_params(), I,
                          distort_model->get_intrinsic_params(), width_,
                          height_, &d_mapx_, &d_mapy_);

  inited_ = true;
  return true;
}

bool UndistortionHandler::Handle(const base::Image8U &src_img,
                                 base::Image8U *dst_img) {
  if (!inited_) {
    return false;
  }

  if (!set_device(device_)) {
    return false;
  }

  return imageRemap(src_img, dst_img, width_, height_, d_mapx_, d_mapy_);
}

bool UndistortionHandler::Release(void) {
  inited_ = false;
  return true;
}

void UndistortionHandler::InitUndistortRectifyMap(
    const Eigen::Matrix3f &camera_model,
    const Eigen::Matrix<float, 8, 1> &distortion, const Eigen::Matrix3f &R,
    const Eigen::Matrix3f &new_camera_model, int width, int height,
    base::Blob<float> *d_mapx, base::Blob<float> *d_mapy) {
  float fx = camera_model(0, 0);
  float fy = camera_model(1, 1);
  float cx = camera_model(0, 2);
  float cy = camera_model(1, 2);
  float nfx = new_camera_model(0, 0);
  float nfy = new_camera_model(1, 1);
  float ncx = new_camera_model(0, 2);
  float ncy = new_camera_model(1, 2);
  float k1 = distortion(0, 0);
  float k2 = distortion(1, 0);
  float p1 = distortion(2, 0);
  float p2 = distortion(3, 0);
  float k3 = distortion(4, 0);
  float k4 = distortion(5, 0);  // add k4,k5,k6 for Rational model
  float k5 = distortion(6, 0);
  float k6 = distortion(7, 0);

  Eigen::Matrix3f Rinv = R.inverse();

  for (int v = 0; v < height_; ++v) {
    float *x_ptr = d_mapx->mutable_cpu_data() + d_mapx->offset(v);
    float *y_ptr = d_mapy->mutable_cpu_data() + d_mapy->offset(v);
    for (int u = 0; u < width_; ++u) {
      Eigen::Matrix<float, 3, 1> xy1;
      xy1 << (static_cast<float>(u) - ncx) / nfx,
          (static_cast<float>(v) - ncy) / nfy, 1;
      auto XYW = Rinv * xy1;
      double nx = XYW(0, 0) / XYW(2, 0);
      double ny = XYW(1, 0) / XYW(2, 0);
      double r_square = nx * nx + ny * ny;
      double r_quad = r_square * r_square;
      double r_sextic = r_quad * r_square;
      double scale = (1 + r_square * k1 + r_quad * k2 + r_sextic * k3) /
                     (1 + r_square * k4 + r_quad * k5 + r_sextic * k6);
      double nnx =
          nx * scale + 2 * p1 * nx * ny + p2 * (r_square + 2 * nx * nx);
      double nny =
          ny * scale + p1 * (r_square + 2 * ny * ny) + 2 * p2 * nx * ny;
      x_ptr[u] = static_cast<float>(nnx * fx + cx);
      y_ptr[u] = static_cast<float>(nny * fy + cy);
    }
  }
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo
