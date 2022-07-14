// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_EXECUTOR_BASE_H_
#define DALI_PIPELINE_EXECUTOR_EXECUTOR_BASE_H_

#include "dali/pipeline/executor/stats.h"

#include <string>

namespace dali {

class OpGraph;
class DeviceWorkspace;

class DLL_PUBLIC ExecutorBase {
 public:
  using ExecutorCallback = std::function<void(void)>;
  DLL_PUBLIC virtual ~ExecutorBase() {}
  DLL_PUBLIC virtual void Build(OpGraph *graph, std::vector<std::string> output_names) = 0;
  DLL_PUBLIC virtual void Init() = 0;
  DLL_PUBLIC virtual void RunCPU() = 0;
  DLL_PUBLIC virtual void RunMixed() = 0;
  DLL_PUBLIC virtual void RunGPU() = 0;
  DLL_PUBLIC virtual void Outputs(DeviceWorkspace *ws) = 0;
  DLL_PUBLIC virtual void ShareOutputs(DeviceWorkspace *ws) = 0;
  DLL_PUBLIC virtual void ReleaseOutputs() = 0;
  DLL_PUBLIC virtual void SetCompletionCallback(ExecutorCallback cb) = 0;
  DLL_PUBLIC virtual void EnableMemoryStats(bool enable_memory_stats = false) = 0;
  DLL_PUBLIC virtual ExecutorMetaMap GetExecutorMeta() = 0;
  DLL_PUBLIC virtual void Shutdown() = 0;

 protected:
  // virtual to allow the TestPruneWholeGraph test in gcc
  virtual void PruneUnusedGraphNodes() = 0;

  template <typename T>
  friend class ExecutorTest;
};

/**
 * @brief Execution parameters of the pipeline.
 */
struct DLL_PUBLIC ExecutionParams {
  int device_id = -1;      /**< id of the GPU to operate on. */
  int num_thread = -1;     /**< the number of threads to use in the prefetch stage. */
  int max_batch_size = -1; /**< the maximum size of the batch that can be produced. */
  int max_num_stream = -1; /**< set an upper limit on the number of cudaStreams that can be
                              allocated by the pipeline. */
  int default_cuda_stream_priority = 0; /**< CUDA stream priority used by DALI. See
                                           `cudaStreamCreateWithPriority` in CUDA documentation */
  size_t bytes_per_sample_hint = 0;     /**< Estimated size of each sample to be processed. */
  bool set_affinity =
      false; /**< indicates whether thread affinity should be configured in the thread pool */
};

/**
 * @brief Configuration for executor mode setup.
 */
struct DLL_PUBLIC ExecutorConfig {
  bool pipelined = true;  /**< whether to allocate the necessary buffers for pipeline execution
                           * between the cpu and gpu portions of the graph. See PipelinedExecutor. */
  bool async = true;      /**< whether to use extra host-threads to enable asynchronous execution
                           * of cpu and gpu work. See AsyncExecutor/AsyncPipelinedExecutor. */
  bool separated = false; /**< whether to use separated queues for pipeline execution */
};


}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_EXECUTOR_BASE_H_