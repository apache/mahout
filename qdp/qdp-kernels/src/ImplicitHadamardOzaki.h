#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

#include "AdaptiveOzaki.h"

namespace ozaki {

class ImplicitHadamardOzakiEngine {
public:
    explicit ImplicitHadamardOzakiEngine(const OzakiConfig& config) : config_(config) {}
    ~ImplicitHadamardOzakiEngine() = default;

    void execute_implicit_hadamard(const double* d_A, double* d_C, int m, int n, int k, double norm_factor, cudaStream_t stream = 0);

private:
    OzakiConfig config_;
};

} // namespace ozaki
