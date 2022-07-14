// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_EXECUTOR_STATS_H_
#define DALI_PIPELINE_EXECUTOR_STATS_H_

#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/api_helper.h"

namespace dali {

struct DLL_PUBLIC ExecutorMeta {
  std::size_t real_size;
  std::size_t max_real_size;
  std::size_t reserved;
  std::size_t max_reserved;
};

using ExecutorMetaMap = std::unordered_map<std::string, std::vector<ExecutorMeta>>;

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_STATS_H_