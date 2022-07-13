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

/// isa, cast, dyncast
namespace dali {
namespace detail {

class EventList {
 public:
  inline EventList() = default;
  inline EventList(int size, EventPool *event_pool) {
    DALI_ENFORCE(event_pool != nullptr);
    for (int i = 0; i < size; ++i) {
      events_.push_back(event_pool->GetEvent());
    }
  }

  inline cudaEvent_t GetEvent(int idx) {
    return events_[idx];
  }

  inline bool empty() const {
    return events_.empty();
  }

 private:
  vector<cudaEvent_t> events_;
};


}  // namespace detail
}  // namespace dali

namespace dali {

// Very simple implementation of llvm style class heirarchy
template <typename To, typename From>
bool isa(From *o) noexcept {
  return std::is_base_of<To, From>::value || To::classof(o);
}

template <typename To, typename From>
bool isa(From &o) noexcept {
  return std::is_base_of<To, From>::value || To::classof(&o);
}

template <typename To, typename From>
bool isa_or_null(From *o) {
  return o != nullptr && isa<To>(o);
}

template <typename To, typename From>
To *cast(From *o) {
  assert(isa<To>(o));
  return static_cast<To *>(o);
}

template <typename To, typename From>
To *dyncast(From *o) {
  return isa<To>(o) ? static_cast<To *>(o) : nullptr;
}

template <typename To, typename From>
To *dyncast_or_null(From *o) {
  if (o == nullptr) {
    return nullptr;
  }
  return dyncast<To>(o);
}

}  // namespace dali

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

  struct MaxSizeHelper {
    template <typename T>
    inline static void GetMaxSizesCont(T &in, size_t &max_out_size, size_t &max_reserved_size) {
      auto out_size = in.nbytes();
      auto reserved_size = in.capacity();
      max_out_size = std::max<size_t>(std::ceil((out_size * 1.0) / in.num_samples()), max_out_size);
      max_reserved_size =
          std::max<size_t>(std::ceil((reserved_size * 1.0) / in.num_samples()), max_reserved_size);
    }

    template <typename T>
    inline static void GetMaxSizesNonCont(T &in, size_t &max_out_size, size_t &max_reserved_size) {
      const auto &nbytes = in._chunks_nbytes();
      const auto &capacity = in._chunks_capacity();
      max_out_size = 0;
      max_reserved_size = 0;
      for (auto &elem : nbytes) {
        max_out_size = std::max(max_out_size, elem);
      }
      for (auto &elem : capacity) {
        max_reserved_size = std::max(max_reserved_size, elem);
      }
    }

    template <typename backend>
    inline static void GetMaxSizes(TensorList<backend> &in, size_t &max_out_size,
                                   size_t &max_reserved_size) {
      GetMaxSizesCont(in, max_out_size, max_reserved_size);
    }

    template <typename backend>
    inline static void GetMaxSizes(TensorVector<backend> &in, size_t &max_out_size,
                                   size_t &max_reserved_size) {
      if (in.IsContiguous()) {
        GetMaxSizesCont(in, max_out_size, max_reserved_size);
      } else {
        GetMaxSizesNonCont(in, max_out_size, max_reserved_size);
      }
    }
  };

  const char *StageStatsPrefix() const noexcept {  // TODO: private
    switch (kind_) {
      case Kind::cpu:
        return "CPU_";
      case Kind::gpu:
        return "GPU_";
      case Kind::cpu2gpu:
        return "MIXED_";
      case Kind::gpu2cpu: {
        assert(false && "Unimplemented");
      }
    }
    return nullptr;
  }

  template <typename Workspace>
  void FillStats(Workspace &ws, std::string op_name) {
    size_t out_size = 0;
    size_t max_out_size = 0;
    size_t reserved_size = 0;
    size_t max_reserved_size = 0;
    std::lock_guard<std::mutex> lck(stats_mutex_);
    auto &stats = stats_[StageStatsPrefix() + op_name];
    stats.resize(ws.NumOutput(), {0, 0});

    for (int i = 0; i < ws.NumOutput(); ++i) {
      out_size = 0;
      max_out_size = 0;
      reserved_size = 0;
      max_reserved_size = 0;
      if (ws.template OutputIsType<CPUBackend>(i)) {
        auto &out = ws.template Output<CPUBackend>(i);
        out_size = out.nbytes();
        reserved_size = out.capacity();
        MaxSizeHelper::GetMaxSizes(out, max_out_size, max_reserved_size);
      } else {
        auto &out = ws.template Output<GPUBackend>(i);
        out_size = out.nbytes();
        reserved_size = out.capacity();
        MaxSizeHelper::GetMaxSizes(out, max_out_size, max_reserved_size);
      }
      stats[i].real_size = std::max(out_size, stats[i].real_size);
      stats[i].max_real_size = std::max(max_out_size, stats[i].max_real_size);
      stats[i].reserved = std::max(reserved_size, stats[i].reserved);
      stats[i].max_reserved = std::max(max_reserved_size, stats[i].max_reserved);
    }
  }

  virtual ~Stage() noexcept = 0;

  ProtectedStatsMap GetStatMapsRef() {
    return std::make_pair(std::ref(stats_), std::ref(stats_mutex_));
  }

 protected:
  Stage(Kind kind) noexcept : kind_(kind) {}
  std::mutex stats_mutex_;
  ExecutorMetaMap stats_;
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
