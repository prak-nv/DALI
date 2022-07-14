// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_
#define DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_

#include <cuda_runtime_api.h>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "dali/core/cuda_error.h"
#include "dali/pipeline/executor/queue_metadata.h"

namespace dali {

// Policy that passes Queueing/Buffering indexes between stages, handling required synchronization
// struct QueuePolicy {
//   // Return sizes of stage queues based of Pipeline arguments
//   static StageQueues GetQueueSizes(QueueSizes init_sizes);
//   // Initialize the policy during Executor::Build();
//   void InitializeQueues(const StageQueues &stage_queue_depths);
//   // Acquire Queue indexes for given stage
//   QueueIdxs AcquireIdxs(OpType stage);
//   // Finish stage and release the indexes. Not called by the last stage, as it "returns" outputs
//   void ReleaseIdxs(OpType stage, QueueIdxs idxs);
//   // Check if acquired indexes are valid
//   bool AreValid(QueueIdxs idxs);
//   // Called by the last stage - mark the Queue idxs as ready to be used as output
//   void QueueOutputIdxs(QueueIdxs idxs);
//   // Get the indexes of ready outputs and mark them as in_use by the user
//   OutputIdxs UseOutputIdxs();
//   // Release currently used output
//   void ReleaseOutputIdxs();
//   // Wake all waiting threads and skip further execution due to stop signaled
//   void SignalStop();
//   // Returns true if we signaled stop previously
//   bool IsStopSignaled();
// };


// Each stage requires ready buffers from previous stage and free buffers from current stage
struct UniformQueuePolicy {
  static const int kInvalidIdx = -1;

  static StageQueues GetQueueSizes(QueueSizes init_sizes) {
    DALI_ENFORCE(init_sizes.cpu_size == init_sizes.gpu_size,
                 "Queue sizes should be equal for UniformQueuePolicy");
    return StageQueues(init_sizes.cpu_size);
  }

  void InitializeQueues(const StageQueues &stage_queue_depths) {
    DALI_ENFORCE(
        stage_queue_depths[OpType::CPU] == stage_queue_depths[OpType::MIXED] &&
            stage_queue_depths[OpType::MIXED] == stage_queue_depths[OpType::GPU],
        "This policy does not support splited queues");

    // All buffers start off as free
    for (int i = 0; i < stage_queue_depths[OpType::CPU]; ++i) {
      free_queue_.push(i);
    }
  }

  QueueIdxs AcquireIdxs(OpType stage) {
    if (!HasPreviousStage(stage)) {
      // Block until there is a free buffer to use
      std::unique_lock<std::mutex> lock(free_mutex_);
      free_cond_.wait(lock, [stage, this]() {
        return !free_queue_.empty() || stage_work_stop_[static_cast<int>(stage)];
      });
      if (stage_work_stop_[static_cast<int>(stage)]) {
        return QueueIdxs{kInvalidIdx};  // We return anything due to exec error
      }
      int queue_idx = free_queue_.front();
      free_queue_.pop();
      return QueueIdxs{queue_idx};
    }

    std::lock_guard<std::mutex> lock(stage_work_mutex_[static_cast<int>(stage)]);
    if (stage_work_stop_[static_cast<int>(stage)]) {
      return QueueIdxs{-1};
    }
    auto &queue = stage_work_queue_[static_cast<int>(stage)];
    assert(!queue.empty());
    auto queue_idx = queue.front();
    queue.pop();
    return QueueIdxs{queue_idx};
  }

  void ReleaseIdxs(OpType stage, QueueIdxs idxs, cudaStream_t = 0) {
    if (idxs[stage] == kInvalidIdx) {
      return;
    }
    if (HasNextStage(stage)) {
      auto next_stage = NextStage(stage);
      std::lock_guard<std::mutex> lock(stage_work_mutex_[static_cast<int>(next_stage)]);
      stage_work_queue_[static_cast<int>(next_stage)].push(idxs[stage]);
    }
  }

  template <OpType op_type>
  bool AreValid(QueueIdxs idxs) {
    OpType prev_stage = op_type;
    bool ret = true;
    while (prev_stage != static_cast<OpType>(-1)) {
      ret = ret && (idxs[prev_stage] != kInvalidIdx);
      prev_stage = PreviousStage(prev_stage);
    }
    return ret;
  }

  void QueueOutputIdxs(QueueIdxs idxs, cudaStream_t = 0) {
    // We have to give up the elements to be occupied
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_queue_.push(idxs[OpType::GPU]);
    }
    ready_cond_.notify_all();
  }

  OutputIdxs UseOutputIdxs() {
    // Block until the work for a batch has been issued.
    // Move the queue id from ready to in_use
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_cond_.wait(lock, [this]() {
      return !ready_queue_.empty() || ready_stop_;
    });
    if (ready_stop_) {
      return OutputIdxs{kInvalidIdx};
    }
    int output_idx = ready_queue_.front();
    ready_queue_.pop();
    in_use_queue_.push(output_idx);
    lock.unlock();
    return OutputIdxs{output_idx};
  }

  void ReleaseOutputIdxs() {
    // Mark the last in-use buffer as free and signal
    // to waiting threads
    // TODO(klecki): in_use_queue should be guarded, but we assume it is used only in synchronous
    // python calls
    if (!in_use_queue_.empty()) {
      {
        std::lock_guard<std::mutex> lock(free_mutex_);
        free_queue_.push(in_use_queue_.front());
        in_use_queue_.pop();
      }
      free_cond_.notify_one();
    }
  }

  void NotifyAll() {
    ready_cond_.notify_all();
    free_cond_.notify_all();
  }

  std::mutex& GetReadyMutex() {
    return ready_mutex_;
  }

  void SignalStop() {
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_stop_ = true;
    }
    for (int i = 0; i < static_cast<int>(OpType::COUNT); ++i) {
      std::lock_guard<std::mutex> l(stage_work_mutex_[i]);
      stage_work_stop_[i] = true;
    }
    NotifyAll();
  }

  bool IsStopSignaled() const {
    // We only need to check the first one, since they're
    // always set in the same time
    return ready_stop_;
  }

 private:
  std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  std::mutex ready_mutex_, free_mutex_;
  std::condition_variable ready_cond_, free_cond_;

  static const int kOpCount = static_cast<int>(OpType::COUNT);
  std::array<std::queue<int>, kOpCount> stage_work_queue_;
  std::array<std::mutex, kOpCount> stage_work_mutex_;
  // We use a dedicated stop flag for every mutex & condition_varialbe pair,
  // so when using them in cond_var predicate,
  // we know the changes are propagated properly and we won't miss a notify.
  std::array<bool, kOpCount> stage_work_stop_ = {{false, false, false}};
  // Used in IsStopSignaled with atomic access, an with mutex for ready_cond_
  std::atomic<bool> ready_stop_ = {false};
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_
