#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
namespace duckdb_onnx {

// 前向声明
class Op;
template<typename T> using TVec = std::vector<T>;
struct InletId;

/// Identifier for a node output in the graph.
///
/// This happens to be a unique identifier of any variable tensor in the graph
/// (as the graph typically connect one single node output to one or several
/// inputs slots)
struct OutletId {
    /// node identifier in the graph
    size_t node;
    /// rank of the input in the node
    size_t slot;

    // 默认构造函数
    OutletId() : node(0), slot(0) {}

    // 带参数的构造函数（相当于Rust的new）
    OutletId(size_t node_id, size_t slot_id) : node(node_id), slot(slot_id) {}

    // 比较运算符
    bool operator==(const OutletId& other) const {
        return node == other.node && slot == other.slot;
    }

    bool operator!=(const OutletId& other) const {
        return !(*this == other);
    }

    bool operator<(const OutletId& other) const {
        return node < other.node || (node == other.node && slot < other.slot);
    }

    bool operator<=(const OutletId& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const OutletId& other) const {
        return !(*this <= other);
    }

    bool operator>=(const OutletId& other) const {
        return !(*this < other);
    }

    // 调试输出
    friend std::ostream& operator<<(std::ostream& os, const OutletId& id) {
        os << id.node << "/" << id.slot << ">";
        return os;
    }
};

// 从整数转换为 OutletId
inline OutletId MakeOutletId(size_t node) {
    return OutletId(node, 0);
}

// 从元组转换为 OutletId
inline OutletId MakeOutletId(std::pair<size_t, size_t> pair) {
    return OutletId(pair.first, pair.second);
}

/// Identifier for a node input in the graph.
struct InletId {
    /// node identifier in the graph
    size_t node;
    /// rank of the input in the node
    size_t slot;

    // 默认构造函数
    InletId() : node(0), slot(0) {}

    // 带参数的构造函数
    InletId(size_t node_id, size_t slot_id) : node(node_id), slot(slot_id) {}

    // 比较运算符
    bool operator==(const InletId& other) const {
        return node == other.node && slot == other.slot;
    }

    bool operator!=(const InletId& other) const {
        return !(*this == other);
    }

    bool operator<(const InletId& other) const {
        return node < other.node || (node == other.node && slot < other.slot);
    }

    bool operator<=(const InletId& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const InletId& other) const {
        return !(*this <= other);
    }

    bool operator>=(const InletId& other) const {
        return !(*this < other);
    }

    // 调试输出
    friend std::ostream& operator<<(std::ostream& os, const InletId& id) {
        os << ">" << id.node << "/" << id.slot;
        return os;
    }
};

/// Information for each outlet of a node
template<typename F>
class Outlet {
public:
    /// the tensor type information
    F fact;
    /// where this outlet is used.
    TVec<InletId> successors;

    // 默认构造函数
    Outlet() = default;

    // 复制构造函数
    Outlet(const Outlet& other) = default;

    // 调试输出
    friend std::ostream& operator<<(std::ostream& os, const Outlet<F>& outlet) {
        os << outlet.fact << " ";
        for (const auto& succ : outlet.successors) {
            os << succ << " ";
        }
        return os;
    }
};

/// A Node in an Model.
///
/// Parameterized by a Fact implementation matching the one used in the
/// model.
template<typename F, typename O>
class Node {
public:
    /// node id in the model
    ///
    /// Caution: this id will not be persistent during networks transformation
    size_t id{};

    /// name of the node
    ///
    /// This will usually come from the importing framework. `tract`
    /// transformation try to maintain the names accross transformations.
    std::string name;

    /// A list of incoming tensors, identified by the node outlet that creates
    /// them.
    std::vector<OutletId> inputs;

    /// The actual operation the node performs.
    O op;

    /// List of ouputs, with their descendant and tensor type information.
    TVec<Outlet<F>> outputs;

    // 默认构造函数
    Node() = default;

    // 复制构造函数
    Node(const Node& other) = default;

    // 复制赋值操作符
    Node& operator=(const Node& other) = default;

    /// Try to downcast the node operation to O.
    template<typename OpType>
    OpType* op_as_mut() {
        return dynamic_cast<OpType*>(static_cast<Op*>(op.as_mut()));
    }


    // 输出为字符串
    friend std::ostream& operator<<(std::ostream& os, const Node<F, O>& node) {
        os << "#" << node.id << " \"" << node.name << "\" " << node.op;
        return os;
    }
};

} // namespace duckdb_onnx

// 为OutletId和InletId提供哈希函数，以便在unordered_map中使用
namespace std {
    template<>
    struct hash<duckdb_onnx::OutletId> {
        size_t operator()(const duckdb_onnx::OutletId& id) const {
            return (hash<size_t>()(id.node) ^ (hash<size_t>()(id.slot) << 1));
        }
    };

    template<>
    struct hash<duckdb_onnx::InletId> {
        size_t operator()(const duckdb_onnx::InletId& id) const {
            return (hash<size_t>()(id.node) ^ (hash<size_t>()(id.slot) << 1));
        }
    };
}