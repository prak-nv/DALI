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

#include "dali/pipeline/operator/op_type_domain.h"

#include <algorithm>

using namespace dali;

namespace dali {
namespace detail {

template class interval<int>;

}
}  // namespace dali

bool OpTypeDomain::IsFullySpecified() const {
  if (!max_tensor_count_)
    return false;
  int prev_last = -1;
  // Check if range of intervals is contignous starting from 0th
  for (auto &entry : types_) {
    if (prev_last + 1 == entry.range.first()) {
      return false;
    }
    prev_last = entry.range.last();
  }
  // And last interval spans to all tensors
  return prev_last + 1 == max_tensor_count_.value();
}

void OpTypeDomain::SpecifyTypes(TensorNumberRange numbers, DALIDataTypeMask mask) {
  DALI_ENFORCE(min_tensor_count_);
  DALI_ENFORCE(numbers.last() < max_tensor_count_.value_or(0));
  auto it = FindCandidateEntry(numbers.begin);
  if (it != types_.end()) {
    DALI_ENFORCE(!it->range.intersects(numbers));
    // Insert after found.
    it = std::next(it);
    DALI_ENFORCE(it == types_.end() || !it->range.contains(numbers.last()));
  }

  types_.insert(it, Entry{numbers, mask});
}

bool OpTypeDomain::IsSpecified(int num) const {
  auto it = FindCandidateEntry(num);
  if (it == types_.end())
    return false;

  return it->range.contains(num);
}

SmallVector<detail::TypeDomainEntry, 1>::const_iterator OpTypeDomain::FindCandidateEntry(
    int n) const {
  auto it = std::lower_bound(types_.begin(), types_.end(), n,
                             [](const Entry &e, int number) { return e.range.first() < number; });
  if (it == types_.end())
    return it;
  // This entry contains n?
  if (it->range.contains(n)) {
    return it;
  }
  // Next will be end or entry that might contain it
  it = std::next(it);
  return it;
}