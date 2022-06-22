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

#ifndef DALI_PIPELINE_OPERATOR_OP_TYPE_DOMAIN_H_
#define DALI_PIPELINE_OPERATOR_OP_TYPE_DOMAIN_H_

#include "dali/core/error_handling.h"
#include "dali/core/optional.h"
#include "dali/core/small_vector.h"
#include "dali/pipeline/data/type_mask.h"

namespace dali {
namespace detail {

template <typename Type>
struct interval {
  using value_type = Type;

  bool empty() const {
    return length > 0;
  }

  value_type first() const {
    DALI_ENFORCE(length > 0);
    return begin;
  }

  value_type last() const {
    DALI_ENFORCE(length > 0);
    return static_cast<value_type>(begin + length - 1);
  }

  bool contains(value_type i) const {
    if (empty())
      return false;
    return i >= first() && i <= last();
  }

  bool contains(interval &other) const {
    if (other.empty())
      return true;
    if (empty())
      return false;
    return contains(other.first()) && contains(other.last());
  }

  bool intersects(const interval &other) const {
    if (empty() || other.empty())
      return false;
    return contains(other.first()) || contains(other.last());
  }

  value_type begin;
  std::size_t length;
};

extern template class interval<int>;

}  // namespace detail

using TensorNumberRange = detail::interval<int>;

namespace detail {

struct TypeDomainEntry {
  using range_type = TensorNumberRange;
  range_type range;
  optional<DALIDataTypeMask> types;
};

}  // namespace detail

struct OpTypeDomain {
  enum DomainKind {
    Input,
    Output
  };

 private:
  using opt_int = dali::optional<int>;

 public:
  using Entry = detail::TypeDomainEntry;

  explicit OpTypeDomain(DomainKind kind) : kind_(kind) {}

  bool IsFullySpecified() const;
  bool IsSpecified(int num) const;

  void SpecifyTypes(TensorNumberRange numbers, DALIDataTypeMask mask);

  std::pair<opt_int, opt_int> getTensorCountLimits() const {
    return std::make_pair(min_tensor_count_, max_tensor_count_);
  }

  opt_int getMinTensorCount() const {
    return min_tensor_count_;
  }

  opt_int getMaxTensorCount() const {
    return max_tensor_count_;
  }

  void setTensorCountLimits(int min, int max) {
    DALI_ENFORCE(min_tensor_count_ == nullopt);
    DALI_ENFORCE(max_tensor_count_ == nullopt);
    min_tensor_count_ = min;
    max_tensor_count_ = max;
  }

  void setTensorCount(int n) {
    DALI_ENFORCE(min_tensor_count_ == nullopt);
    DALI_ENFORCE(max_tensor_count_ == nullopt);
    min_tensor_count_ = n;
    max_tensor_count_ = n;
  }

 private:
  SmallVector<Entry, 1>::const_iterator FindCandidateEntry(int n) const;
  // NB: Sorted by intervals
  SmallVector<Entry, 1> types_;
  opt_int min_tensor_count_;
  opt_int max_tensor_count_;
  DomainKind kind_;
};

}  // namespace dali

#endif