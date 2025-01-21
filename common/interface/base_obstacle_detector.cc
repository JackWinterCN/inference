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
#include "common/interface/base_obstacle_detector.h"

#include <map>
#include <vector>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "common/inference/inference_factory.h"
#include "common/inference/model_util.h"

namespace apollo {
namespace perception {
namespace camera {

bool BaseObstacleDetector::InitNetwork(const common::ModelInfo& model_info,
                                       const std::string& model_root) {
  // Network files
  std::string proto_file = cyber::common::GetAbsolutePath(
      model_root, model_info.proto_file().file());
  std::string weight_file = cyber::common::GetAbsolutePath(
      model_root, model_info.weight_file().file());

  // Network input and output names
  std::vector<std::string> input_names =
      inference::GetBlobNames(model_info.inputs());
  std::vector<std::string> output_names =
      inference::GetBlobNames(model_info.outputs());

  // Network type
  const auto &framework = model_info.framework();
  std::string plugin_name = model_info.infer_plugin();
  static const std::string class_namespace =
      "apollo::perception::inference::";
  if (model_info.has_infer_plugin() && !plugin_name.empty()) {
    plugin_name = class_namespace + plugin_name;
    net_ = apollo::cyber::plugin_manager::PluginManager::Instance()
             ->CreateInstance<inference::Inference>(plugin_name);
    net_->set_model_info(proto_file, input_names, output_names);
    AINFO << "net load plugin success: " << plugin_name;
  } else {
    net_.reset(inference::CreateInferenceByName(framework, proto_file,
                                                weight_file, output_names,
                                                input_names, model_root));
  }

  ACHECK(net_ != nullptr);
  net_->set_gpu_id(gpu_id_);

  std::map<std::string, std::vector<int>> shape_map;
  inference::AddShape(&shape_map, model_info.inputs());
  inference::AddShape(&shape_map, model_info.outputs());

  if (!net_->Init(shape_map)) {
    AERROR << model_info.name() << "init failed!";
    return false;
  }
  return true;
}



// class BaseObstacleDetectorRegisterer { typedef ::apollo::perception::lib::Any Any; typedef ::apollo::perception::lib::FactoryMap FactoryMap; public: static BaseObstacleDetector *GetInstanceByName(const ::std::string &name) { FactoryMap &map = ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"]; FactoryMap::iterator iter = map.find(name); if (iter == map.end()) { for (auto c : map) { google::LogMessage("/home/zhito/workspace/Apollo/apollo/inference/common/interface/base_obstacle_detector.cc", 75, google::ERROR).stream() << "[" << apollo::cyber::binary::GetName().c_str() << "]" << "Instance:" << c.first; } google::LogMessage("/home/zhito/workspace/Apollo/apollo/inference/common/interface/base_obstacle_detector.cc", 75, google::ERROR).stream() << "[" << apollo::cyber::binary::GetName().c_str() << "]" << "Get instance " << name << " failed."; return nullptr; } Any object = iter->second->NewInstance(); return *(object.AnyCast<BaseObstacleDetector *>()); } static std::vector<BaseObstacleDetector *> GetAllInstances() { std::vector<BaseObstacleDetector *> instances; FactoryMap &map = ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"]; instances.reserve(map.size()); for (auto item : map) { Any object = item.second->NewInstance(); instances.push_back(*(object.AnyCast<BaseObstacleDetector *>())); } return instances; } static const ::std::string GetUniqInstanceName() { FactoryMap &map = ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"]; while (google::_Check_string* _result = google::Check_EQImpl( google::GetReferenceableValue(map.size()), google::GetReferenceableValue(1U), "map.size()" " " "==" " " "1U")) google::LogMessageFatal("/home/zhito/workspace/Apollo/apollo/inference/common/interface/base_obstacle_detector.cc", 75, google::CheckOpString(_result)).stream() << map.size(); return map.begin()->first; } static BaseObstacleDetector *GetUniqInstance() { FactoryMap &map = ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"]; while (google::_Check_string* _result = google::Check_EQImpl( google::GetReferenceableValue(map.size()), google::GetReferenceableValue(1U), "map.size()" " " "==" " " "1U")) google::LogMessageFatal("/home/zhito/workspace/Apollo/apollo/inference/common/interface/base_obstacle_detector.cc", 75, google::CheckOpString(_result)).stream() << map.size(); Any object = map.begin()->second->NewInstance(); return *(object.AnyCast<BaseObstacleDetector *>()); } static bool IsValid(const ::std::string &name) { FactoryMap &map = ::apollo::perception::lib::GlobalFactoryMap()["BaseObstacleDetector"]; return map.find(name) != map.end(); } };

}  // namespace camera
}  // namespace perception
}  // namespace apollo
