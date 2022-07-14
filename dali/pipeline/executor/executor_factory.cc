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

#include "dali/pipeline/executor/executor_factory.h"

#include "dali/pipeline/executor/async_pipelined_executor.h"
#include "dali/pipeline/executor/executor.h"
#include "dali/pipeline/executor/pipelined_executor.h"

using namespace dali;

template <typename... Ts>
static std::unique_ptr<ExecutorBase> GetExecutorImpl(bool pipelined, bool async,
                                                     Ts&&... args) {
  if (async && pipelined) {
    return std::unique_ptr<ExecutorBase>{new AsyncPipelinedExecutor(std::forward<Ts>(args)...)};
  } else if (!async && pipelined) {
    return std::unique_ptr<ExecutorBase>{new PipelinedExecutor(std::forward<Ts>(args)...)};
  } else if (!async && !pipelined) {
    return std::unique_ptr<ExecutorBase>{new SimpleExecutor(std::forward<Ts>(args)...)};
  }
  std::stringstream error;
  error << std::boolalpha;
  error << "No supported executor selected for pipelined = " << pipelined
        << ", async = " << async << std::endl;
  DALI_FAIL(error.str());
}

std::unique_ptr<ExecutorBase> dali::GetExecutor(ExecutorConfig config, ExecutionParams params,
                                                QueueSizes prefetch_queue_depth) {
  if (config.separated) {
    prefetch_queue_depth = QueueSizes(prefetch_queue_depth.cpu_size, prefetch_queue_depth.cpu_size);
    // TODO: issue warning
  }
  return GetExecutorImpl(config.pipelined, config.async, params,
                         prefetch_queue_depth);
}
