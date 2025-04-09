#pragma once

#include "duckdb-onnx/core/common.hpp"
#include "onnx.proto3.pb.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace duckdb_onnx {
namespace pb {
class ModelProto;
class GraphProto;
class NodeProto;
} // namespace pb

class Onnx;
class ParsingContext {
public:
	int64_t onnx_operator_set_version {};
	const Onnx *framework {};
	const pb::ModelProto *model {};
	std::vector<const pb::GraphProto *> parent_graphs;
	std::optional<std::string> model_dir;

	ParsingContext() = default;
	ParsingContext(int64_t version, const Onnx *fw, const pb::ModelProto *m, std::vector<const pb::GraphProto *> graphs,
	               const std::string *dir = nullptr
	               // InferenceModel temp = InferenceModel()
	               )
	    : onnx_operator_set_version(version), framework(fw), model(m), parent_graphs(std::move(graphs)),
	      model_dir(dir ? std::optional<std::string>(*dir) : std::nullopt) {
	}
};

class ModelDataResolver {
public:
	virtual ~ModelDataResolver() = default;
};

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
	Onnx() : use_output_shapes(false), ignore_output_types(false) {
	}

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
//   OnnxOpRegister() = default;
//   void Register(const std::string& op_name, const OpBuilder& builder) {
//     op_builders[op_name] = builder;
//   }
//   static OpBuilder* Find(const std::string& op_name) {
//     auto it = op_builders.find(op_name);
//     if (it != op_builders.end()) {
//       return &(it->second);
//     }
//     return nullptr;
//   }
// };

} // namespace duckdb_onnx
