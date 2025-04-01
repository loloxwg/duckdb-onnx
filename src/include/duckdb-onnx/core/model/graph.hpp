#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <iostream>
#include "duckdb-onnx/value.h"
#include "duckdb-onnx/tensor.h"
#include "duckdb-onnx/core/model/node.hpp"

namespace duckdb_onnx {

class Op;
struct OutletId;
class Tensor;
/// Main model class
///
/// Parameterized by a Fact class.
template<typename F, typename O>
class Graph {
public:
    /// all nodes in the model
    std::vector<Node<F, O>> nodes{};
    /// model inputs
    std::vector<OutletId> inputs;
    /// model outputs
    std::vector<OutletId> outputs;
    /// outlet labels
    std::unordered_map<OutletId, std::string> outlet_labels;
    /// model properties
    std::unordered_map<std::string, std::shared_ptr<Tensor>> properties;
    /// symbol scope, including table

    // 可能需要添加构造函数、析构函数和其他必要的方法
    Graph() = default;
    Graph(const Graph& other) = default;
    Graph& operator=(const Graph& other) = default;
};

} // namespace duckdb_onnx