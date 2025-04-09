#pragma once
#include <vector>
namespace duckdb_onnx {
enum DatumType {
	Bool,
	U8,
	U16,
	U32,
	U64,
	I8,
	I16,
	I32,
	I64,
	F16,
	F32,
	F64,
	TDim,
	Blob,
	String,
};

class Blob {
	char *data;
};

class Tensor {
	DatumType dt;
	std::vector<int> shape;
	std::vector<int> strides;
	int len;
	class Blob data;
};
} // namespace duckdb_onnx
