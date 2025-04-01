#pragma once
#include <string>


namespace duckdb_onnx {
class Error {
public:
    explicit Error(std::string msg) : message_(std::move(msg)) {}
    virtual ~Error() = default;

    const std::string& what() const { return message_; }

private:
    std::string message_;
};
template<typename T>
class Result {
    union Data {
        T value;
        Error error;
        Data() {}
        ~Data() {}
    };

public:
    explicit Result(const T& value) : is_ok_(true) {
        new(&data_.value) T(value);
    }

    explicit Result(T&& value) : is_ok_(true) {
        new(&data_.value) T(std::move(value));
    }

    explicit Result(const Error& error) : is_ok_(false) {
        new(&data_.error) Error(error);
    }

    explicit Result(Error&& error) : is_ok_(false) {
        new(&data_.error) Error(std::move(error));
    }

    Result(const Result& other) : is_ok_(other.is_ok_) {
        if (is_ok_) {
            new(&data_.value) T(other.data_.value);
        } else {
            new(&data_.error) Error(other.data_.error);
        }
    }

    Result(Result&& other)  noexcept : is_ok_(other.is_ok_) {
        if (is_ok_) {
            new(&data_.value) T(std::move(other.data_.value));
        } else {
            new(&data_.error) Error(std::move(other.data_.error));
        }
    }

    // 析构函数 - 手动调用适当的析构函数
    ~Result() {
        if (is_ok_) {
            data_.value.~T();
        } else {
            data_.error.~Error();
        }
    }

    Result& operator=(const Result& other) {
        if (this != &other) {
            if (is_ok_) {
                data_.value.~T();
            } else {
                data_.error.~Error();
            }
            is_ok_ = other.is_ok_;
            if (is_ok_) {
                new(&data_.value) T(other.data_.value);
            } else {
                new(&data_.error) Error(other.data_.error);
            }
        }
        return *this;
    }

    // 移动赋值运算符
    Result& operator=(Result&& other) {
        if (this != &other) {
            if (is_ok_) {
                data_.value.~T();
            } else {
                data_.error.~Error();
            }

            is_ok_ = other.is_ok_;

            if (is_ok_) {
                new(&data_.value) T(std::move(other.data_.value));
            } else {
                new(&data_.error) Error(std::move(other.data_.error));
            }
        }
        return *this;
    }

    bool is_ok() const {
        return is_ok_;
    }

    bool is_err() const {
        return !is_ok_;
    }

    const T& value() const {
        if (!is_ok_) {
            throw std::runtime_error(data_.error.what());
        }
        return data_.value;
    }

    T& value() {
        if (!is_ok_) {
            throw std::runtime_error(data_.error.what());
        }
        return data_.value;
    }

    T&& value_move() {
        if (!is_ok_) {
            throw std::runtime_error(data_.error.what());
        }
        return std::move(data_.value);
    }

    const Error& error() const {
        if (is_ok_) {
            throw std::runtime_error("Result contains a value, not an error");
        }
        return data_.error;
    }

    static Result Ok(T value) {
        return Result(std::move(value));
    }

    static Result Err(std::string message) {
        return Result(Error(std::move(message)));
    }

    template<typename F, typename R = typename std::result_of<F(T)>::type>
    Result<R> map(F&& f) const {
        if (is_ok_) {
            try {
                return Result<R>::Ok(f(value()));
            } catch (const std::exception& e) {
                return Result<R>::Err(e.what());
            }
        }
        return Result<R>::Err(error().what());
    }

private:
    bool is_ok_;
    Data data_;
};

template<typename T>
using TractResult = Result<T>;

template<typename T>
TractResult<T> Ok(T value) {
    return TractResult<T>::Ok(std::move(value));
}

template<typename T>
TractResult<T> Err(std::string message) {
    return TractResult<T>::Err(std::move(message));
}

template<typename F, typename... Args>
auto try_catch(F&& f, Args&&... args)
    -> TractResult<typename std::result_of<F(Args...)>::type> {
    using ResultType = typename std::result_of<F(Args...)>::type;

    try {
        return Ok<ResultType>(f(std::forward<Args>(args)...));
    } catch (const std::exception& e) {
        return Err<ResultType>(e.what());
    }
}

}