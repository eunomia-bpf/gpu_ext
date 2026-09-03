#pragma once
// Numerical instrumentation shared by native, original GPreempt, and BPF cells.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpreempt_artifact {
inline void validate_config(int role, uint64_t preprocess_us) {
    if ((role != 0 && role != 1) || preprocess_us <= 100)
        throw std::runtime_error("GPreempt comparison requires role 0/1 and preprocess_time > 100 us");
}

inline void initialize_input(void *input, size_t bytes) {
    if (!input || !bytes || bytes % sizeof(float))
        throw std::runtime_error("GPreempt model comparison requires a nonempty FP32 input");
    auto *values = static_cast<float *>(input);
    for (size_t i = 0; i < bytes / sizeof(float); ++i)
        values[i] = (static_cast<int>(i % 257) - 128) / 128.0f;
}

class OutputCheck {
public:
    static constexpr double atol = 1e-6, rtol = 1e-4;

    void initialize(const std::string &name, const std::string &reference_path,
                    void *input, size_t input_bytes, size_t output_bytes) {
        initialize_input(input, input_bytes);
        if (!output_bytes || output_bytes % sizeof(float))
            throw std::runtime_error("GPreempt model comparison requires a nonempty FP32 output");
        if (reference.empty()) {
            for (char c : name)
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '_' || c == '-'))
                    throw std::runtime_error("validation name must use letters, digits, '_' or '-'");
            label = name;
            std::ifstream source(reference_path, std::ios::binary | std::ios::ate);
            if (!source || source.tellg() != static_cast<std::streamoff>(output_bytes))
                throw std::runtime_error("missing or incorrectly sized native reference: " + reference_path);
            reference.resize(output_bytes / sizeof(float));
            source.seekg(0);
            source.read(reinterpret_cast<char *>(reference.data()), output_bytes);
            if (!source)
                throw std::runtime_error("incomplete native reference: " + reference_path);
            for (float value : reference)
                if (!std::isfinite(value))
                    throw std::runtime_error("nonfinite native reference");
        } else if (reference.size() * sizeof(float) != output_bytes || label != name) {
            throw std::runtime_error("reference dimensions changed during context reinitialization");
        }
    }

    void begin_timed() { timed = true; }

    void check(const void *output, size_t bytes) {
        if (!output || reference.empty() || bytes != reference.size() * sizeof(float))
            throw std::runtime_error("output validation called with incorrect dimensions");
        const auto *values = static_cast<const float *>(output);
        for (size_t i = 0; i < reference.size(); ++i) {
            double error = std::abs(static_cast<double>(values[i]) - reference[i]);
            if (!std::isfinite(values[i]) || error > atol + rtol * std::abs(reference[i])) {
                std::fprintf(stderr, "GPreempt numerical mismatch: task=%s index=%zu observed=%.9g expected=%.9g\n",
                             label.c_str(), i, values[i], reference[i]);
                throw std::runtime_error("GPreempt output differs from isolated native reference");
            }
            max_absolute_error = std::max(max_absolute_error, error);
        }
        ++checked;
        if (timed) ++timed_checked;
    }

    uint64_t count() const { return checked; }
    uint64_t timed_count() const { return timed_checked; }

    ~OutputCheck() {
        if (!label.empty())
            std::fprintf(stderr, "GPREEMPT_VALIDATION {\"task\":\"%s\",\"checked\":%llu,"
                         "\"timed_checked\":%llu,\"max_absolute_error\":%.9g,"
                         "\"atol\":%.9g,\"rtol\":%.9g}\n", label.c_str(),
                         static_cast<unsigned long long>(checked),
                         static_cast<unsigned long long>(timed_checked), max_absolute_error, atol, rtol);
    }
private:
    std::string label;
    std::vector<float> reference;
    bool timed = false;
    uint64_t checked = 0, timed_checked = 0;
    double max_absolute_error = 0;
};
} // namespace gpreempt_artifact
