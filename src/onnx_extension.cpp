#define DUCKDB_EXTENSION_MAIN

#include "onnx_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <onnxruntime_cxx_api.h>
#include <openssl/opensslv.h>

namespace duckdb {

struct TensorOutput {
  std::vector<int64_t> shape;
  std::vector<float> values;
};

TensorOutput ConvertToTensorOutput(Ort::Value &tensor) {

  try {
    TensorOutput result;
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    result.shape = tensor_info.GetShape();

    auto *data = tensor.GetTensorMutableData<float>();
    size_t total_size = tensor_info.GetElementCount();
    result.values.assign(data, data + total_size);

    return result;

  } catch (const Ort::Exception &e) {
    throw std::runtime_error("Convert tensor ONNX Runtime error：" +
                             std::string(e.what()));
  }
}

inline void OnnxScalarFun(DataChunk &args, ExpressionState &state,
                          Vector &result) {
  auto &str_vector = args.data[0];
  auto &struct_vector = args.data[1];

  vector<unique_ptr<Vector>> &source_child =
      StructVector::GetEntries(struct_vector);

  const auto path = str_vector.GetValue(0).ToString();

  Vector &shape_vector = *source_child.front();
  Vector &value_vector = *source_child.back();

  idx_t shape_list_size = ListVector::GetListSize(shape_vector);
  idx_t value_list_size = ListVector::GetListSize(value_vector);

  std::vector<int64_t> input_shape;
  input_shape.reserve(shape_list_size);
  for (int idx = 0; idx < shape_list_size; idx++) {
    input_shape.push_back(
        ListVector::GetEntry(shape_vector).GetValue(idx).GetValue<int64_t>());
  }

  std::vector<float> input_data;
  input_data.reserve(value_list_size);
  for (int idx = 0; idx < value_list_size; idx++) {
    input_data.push_back(
        ListVector::GetEntry(value_vector).GetValue(idx).GetValue<float>());
  }

  try {
    Ort::Env ort_env;
    Ort::Session session{ort_env, (path.c_str()), Ort::SessionOptions{nullptr}};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(),
        input_shape.size());

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> output_shape = tensor_info.GetShape();

    size_t output_size = 1;
    for (int64_t dim : output_shape) {
      if (dim < 0) {
        dim = input_shape[0];
      }
      output_size *= dim;
    }

    std::vector<float> output_data(output_size);
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output_data.data(), output_size, output_shape.data(),
        output_shape.size());

    const char *input_names[] = {"X"};
    const char *output_names[] = {"Y"};

    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                output_names, &output_tensor, 1);

    std::vector<Value> shape_vl_list;
    shape_vl_list.reserve(output_shape.size());
    for (int64_t dim : output_shape) {
      shape_vl_list.emplace_back(dim);
    }

    auto shape_val_list = Value::LIST(std::move(shape_vl_list));
    std::vector<Value> value_vl_list;
    value_vl_list.reserve(output_data.size());
    for (float val : output_data) {
      value_vl_list.emplace_back(val);
    }
    auto value_val_list = Value::LIST(std::move(value_vl_list));

    child_list_t<Value> struct_vl_list;
    struct_vl_list.push_back(make_pair("shape", shape_val_list));
    struct_vl_list.push_back(make_pair("value", value_val_list));
    auto max_struct_val_list = Value::STRUCT(std::move(struct_vl_list));

    result.SetValue(0, max_struct_val_list);
    result.SetVectorType(VectorType::CONSTANT_VECTOR);

  } catch (...) {
    throw std::runtime_error("Onnxruntime error");
  }
}

unique_ptr<FunctionData>
OnnxBindFunction(ClientContext &, ScalarFunction &bound_function,
                 vector<unique_ptr<Expression>> &arguments) {
  switch (arguments[1]->return_type.id()) {
  case LogicalTypeId::UNKNOWN:
    throw ParameterNotResolvedException();
  case LogicalTypeId::LIST:
    break;
  default:
    throw NotImplementedException(
        "onnx(string, list) requires a list as parameter");
  }
  bound_function.arguments[1] = arguments[1]->return_type;
  bound_function.return_type = arguments[1]->return_type;
  return nullptr;
}

static void LoadInternal(DatabaseInstance &instance) {
  // Register a scalar function
  auto int_list_type = LogicalType::LIST(LogicalType::INTEGER);
  auto float_list_type = LogicalType::LIST(LogicalType::FLOAT);
  child_list_t<LogicalType> struct_list_type_list;
  struct_list_type_list.push_back(make_pair("shape", int_list_type));
  struct_list_type_list.push_back(make_pair("value", float_list_type));
  auto struct_list_type = LogicalType::STRUCT(struct_list_type_list);

  auto onnx_scalar_function = duckdb::ScalarFunction(
      "onnx", {}, struct_list_type, OnnxScalarFun, nullptr, nullptr, nullptr,
      nullptr, duckdb::LogicalType::ANY);

  ExtensionUtil::RegisterFunction(instance, onnx_scalar_function);
}

void OnnxExtension::Load(DuckDB &db) { LoadInternal(*db.instance); }
std::string OnnxExtension::Name() { return "onnx"; }

std::string OnnxExtension::Version() const {
#ifdef EXT_VERSION_ONNX
  return EXT_VERSION_ONNX;
#else
  return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void onnx_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<duckdb::OnnxExtension>();
}

DUCKDB_EXTENSION_API const char *onnx_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
