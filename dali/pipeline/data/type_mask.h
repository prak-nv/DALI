#ifndef GUARD_
#define GUARD_

#include "dali/core/bitmask.h"
#include "dali/pipeline/data/types.h"

#include <initializer_list>

namespace dali {

struct DALIDataTypeMask final : private bitmask {
  using EC = enumerator_count<DALIDataType>;
  static_assert(HasEnumeratorCount<DALIDataType>(), "");

 private:
  using base = bitmask;
  constexpr ptrdiff_t bit_idx(DALIDataType dtype) const {
    assert(static_cast<ptrdiff_t>(dtype) >= 0);
    assert(static_cast<ptrdiff_t>(dtype) < bitmask::ssize());
    return static_cast<ptrdiff_t>(dtype);
  }

 public:
  DALIDataTypeMask() : bitmask() {
    // NB: unused 0th bit for enumerator values
    base::resize(EC{} + 1);
  }

  DALIDataTypeMask(std::initializer_list<DALIDataType> lst) : DALIDataTypeMask() {
    add(lst);
  }

  /* explicit(false) */ DALIDataTypeMask(DALIDataType dtype) : DALIDataTypeMask() {
    add(dtype);
  }

  void add(DALIDataType dtype) {
    base::operator[](bit_idx(dtype)) |= true;
  }

  void add(std::initializer_list<DALIDataType> lst) {
    for (auto v : lst) {
      add(v);
    };
  }

  void remove(DALIDataType dtype) {
    base::operator[](bit_idx(dtype)) &= false;
  }

  void remove(std::initializer_list<DALIDataType> lst) {
    for (auto v : lst) {
      remove(v);
    };
  }

  bool has(DALIDataType dtype) const {
    return base::operator[](bit_idx(dtype));
  }

  auto operator[] (DALIDataType dtype) {
    return base::operator[](bit_idx(dtype));
  }

  auto operator[] (DALIDataType dtype) const {
    return base::operator[](bit_idx(dtype));
  }

  auto values() {
    dali::SmallVector<DALIDataType, enumerator_count<DALIDataType>{}> ret;
    ptrdiff_t v = -1;
    while ((v = bitmask::find(true, v+1)) < base::ssize()) {
      ret.push_back(static_cast<DALIDataType>(v));
    }
    return ret;
  }
};

}
#endif