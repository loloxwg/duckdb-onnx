#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "duckdb-onnx/error.h"

namespace duckdb_onnx {

enum class Validation {
  Random,  // 输出是随机的
  Rounding,  // 实现可能引入舍入误差
  Accurate,  // 实现必须准确
};

class TValue;
// EvalOp 基础接口类
class EvalOp {
public:
  virtual ~EvalOp() = default;
  // EvalOp 的方法...
  virtual TractResult<std::vector<TValue>> eval(
      const std::vector<TValue>& inputs) const = 0;
};



// Op 基础接口类
class Op : public EvalOp {
public:
  virtual ~Op() = default;

  // 返回操作符名称
  virtual std::string name() const = 0;

  // 返回验证类型
  virtual Validation validation() const {
    return Validation::Accurate;
  }

  // 比较两个操作符是否相同
  virtual bool same_as(const Op* other) const {
    return false;
  }
  // 调试输出方法（对应 Debug trait）
  virtual void debug_print(std::ostream& os) const {
    os << "Op(" << name() << ")";
  }

  // 克隆方法（对应 DynClone trait）
  virtual std::unique_ptr<Op> clone() const = 0;
};


// 重载输出操作符，用于调试
inline std::ostream& operator<<(std::ostream& os, const Op& op) {
  op.debug_print(os);
  return os;
}



}