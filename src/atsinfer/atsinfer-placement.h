#ifndef ATSINFER_PLACEMENT_H
#define ATSINFER_PLACEMENT_H

#include "atsinfer-profiler.h"
#include <vector>
#include <string>
#include <unordered_map>

enum class ATSInferBackend {
    CPU,
    GPU
};

struct atsinfer_placement_decision {
    std::unordered_map<std::string, ATSInferBackend> placement;
    size_t total_vram_used_bytes;
    float expected_total_latency_ms;
};

// Algoritmo 1: Static Tensor Placement Solver (Dense & MoE)
atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model
);

#endif // ATSINFER_PLACEMENT_H
