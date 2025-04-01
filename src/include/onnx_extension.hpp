#pragma once
#include "duckdb-onnx/core/common.hpp"

namespace duckdb {

class OnnxExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
        std::string Version() const override;
};

} // namespace duckdb
