#pragma once

#include "duckdb-onnx/core/common.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include "onnx.proto3.pb.h"

namespace duckdb_onnx {
namespace pb {
class ModelProto;
class GraphProto;
class NodeProto;
}

class Onnx;
class ParsingContext {
public:
  int64_t onnx_operator_set_version{};
  const Onnx *framework{};
  const pb::ModelProto *model{};
  std::vector<const pb::GraphProto *> parent_graphs;
  std::optional<std::string> model_dir;
  // InferenceModel
  //     template_model; // 注意：在C++中"template"是关键字，所以改名为template_model

  // 构造函数
  ParsingContext() = default;

  // 可选：添加构造函数以简化初始化
  ParsingContext(int64_t version, const Onnx *fw, const pb::ModelProto *m,
                 std::vector<const pb::GraphProto *> graphs,
                 const std::string *dir = nullptr
                 // InferenceModel temp = InferenceModel()
                 )
      : onnx_operator_set_version(version), framework(fw), model(m),
        parent_graphs(std::move(graphs)),
        model_dir(dir ? std::optional<std::string>(*dir) : std::nullopt){}
};


class ModelDataResolver {
public:
  virtual ~ModelDataResolver() = default;
  // 接口方法定义...
};

// 使用 std::function 定义 OpBuilder 类型
// using OpBuilder = std::function<
//     std::pair<std::unique_ptr<InferenceOp>, std::vector<std::string>>(
//         const ParsingContext *, const pb::NodeProto *)>;

// typedef std::unordered_map<std::string, OpBuilder> OnnxOpRegister;

class Onnx {
public:
  // OnnxOpRegister op_register;
  bool use_output_shapes;
  bool ignore_output_types;
  std::shared_ptr<ModelDataResolver> provider;

  // 构造函数
  Onnx() : use_output_shapes(false), ignore_output_types(false) {}

  // 复制构造函数 (对应 Rust 中的 Clone trait)
  Onnx(const Onnx &other) = default;

  // 复制赋值运算符
  Onnx &operator=(const Onnx &other) = default;
};

// using OpBuilder = std::function<
//     std::pair<std::unique_ptr<InferenceOp>, std::vector<std::string>>(
//         const ParsingContext *, const pb::NodeProto *)>;

// class OnnxOpRegister {
// public:
//   std::unordered_map<std::string, OpBuilder> op_builders{};
//
//   // 构造函数
//   OnnxOpRegister() = default;
//
//   // 添加操作构建器
//   void Register(const std::string& op_name, const OpBuilder& builder) {
//     op_builders[op_name] = builder;
//   }
//
//   // 查找操作构建器
//   static OpBuilder* Find(const std::string& op_name) {
//     auto it = op_builders.find(op_name);
//     if (it != op_builders.end()) {
//       return &(it->second);
//     }
//     return nullptr;
//   }
// };




} // namespace duckdb_onnx
