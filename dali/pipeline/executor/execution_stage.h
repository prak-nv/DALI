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


#ifndef DALI_PIPELINE_EXECUTOR_EXECUTION_STAGE_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTION_STAGE_H_

#include <cmath>
#include <mutex>
#include <utility>

#include "dali/core/common.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/graph/op_graph_storage.h"
#include "dali/pipeline/util/event_pool.h"

namespace dali {


struct DLL_PUBLIC ExecutorMeta {
  size_t real_size;
  size_t max_real_size;
  size_t reserved;
  size_t max_reserved;
};

template <typename T1, typename T2>
using ref_pair = std::pair<std::reference_wrapper<T1>, std::reference_wrapper<T2>>;

using ExecutorMetaMap = std::unordered_map<std::string, std::vector<ExecutorMeta>>;
using ProtectedStatsMap = ref_pair<ExecutorMetaMap, std::mutex>;

struct Stage {
  enum class Kind {
    cpu,
    gpu,
    cpu2gpu,
    gpu2cpu
  };

  static constexpr inline bool UsesGPU(Kind k) noexcept {
    return k != Kind::cpu;
  }

  Stage(const Stage &) = delete;
  Stage &operator=(const Stage &) = delete;

  static inline bool classof(Stage *) {
    return true;
  }

  // TODO(prak) Kill...
  OpType GetOpType() const noexcept {
    switch (kind_) {
      case Kind::cpu:
        return OpType::CPU;
      case Kind::gpu:
        return OpType::GPU;
      default:
        return OpType::MIXED;
    }
  }

  auto GetKind() noexcept {
    return kind_;
  }

  bool UsesGPU() noexcept {
    return UsesGPU(kind_);
  }

  virtual ~Stage() noexcept = 0;

 protected:
  Stage(Kind kind) noexcept : kind_(kind) {}
  Kind kind_;
};

inline Stage::~Stage() noexcept = default;

struct DeviceStageBase : public Stage {
  DeviceStageBase(Stage::Kind k) : Stage(k) {
    DALI_ENFORCE(k != Kind::cpu);
  }

  static inline bool classof(Stage *s) {
    return s->GetKind() != Kind::cpu;
  }

};

struct GPUStage final : public DeviceStageBase {
  GPUStage() : DeviceStageBase(Kind::gpu) {}
  static inline bool classof(Stage *s) {
    return s->GetKind() == Stage::Kind::gpu;
  }
};

struct CPU2GPUStage final : public DeviceStageBase {
  CPU2GPUStage() : DeviceStageBase(Kind::cpu2gpu) {}
  static inline bool classof(Stage *s) {
    return s->GetKind() == Stage::Kind::cpu2gpu;
  }
  MixedOpEventMap mixed_op_events_;
};

struct CPUStage final : public Stage {
  CPUStage() : Stage(Kind::cpu) {}
  static inline bool classof(Stage *s) {
    return s->GetKind() == Stage::Kind::cpu;
  }
};

using stage_ptr_t = std::unique_ptr<Stage>;
// For now simple array of stages
using ExecutionPlan = std::array<stage_ptr_t, 3>;

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTION_STAGE_H_
