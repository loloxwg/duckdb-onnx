#pragma once
#include "duckdb-onnx/core/model/node.hpp"
#include "duckdb-onnx/tensor.h"
#include "duckdb-onnx/value.h"
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb_onnx {

class Op;
struct OutletId;
class Tensor;
/// Main model class
///
/// Parameterized by a Fact class.
template <typename F, typename O> class Graph {
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

  Graph() = default;
  Graph(const Graph &other) = default;
  Graph &operator=(const Graph &other) = default;
};

} // namespace duckdb_onnx