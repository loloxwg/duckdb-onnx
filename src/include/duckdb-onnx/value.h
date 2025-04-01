#pragma once
#include <memory>
#include "tensor.h"


namespace duckdb_onnx {

class TValue {
public:
    TValue() = default;
    TValue(const TValue& other) = default;
    TValue(TValue&& other) = default;
    TValue& operator=(const TValue& other) = default;
    TValue& operator=(TValue&& other) = default;
    ~TValue() = default;

    bool is_exclusive() const {
        if (is_var_) {
            return tensor_.use_count() == 1;
        }
        return false;
    }

    const std::shared_ptr<Tensor>* as_arc_tensor() const {
        if (!is_var_) {
            return &tensor_;
        }
        return nullptr;
    }

    const Tensor& operator*() const {
        return *tensor_;
    }

    const Tensor* operator->() const {
        return tensor_.get();
    }

    std::shared_ptr<Tensor> tensor_;
    bool is_var_{}; // true 表示 Var，false 表示 Const
};

}
