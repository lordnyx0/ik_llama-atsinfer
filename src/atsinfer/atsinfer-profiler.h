#ifndef ATSINFER_PROFILER_H
#define ATSINFER_PROFILER_H

#include "ggml.h"
#include <string>
#include <vector>
#include <unordered_map>

struct atsinfer_tensor_profile {
    std::string tensor_name;
    size_t size_bytes;         // s_i
    float exec_time_cpu_ms;    // t_i^c
    float exec_time_gpu_ms;    // t_i^g
    float latency_reduction;   // r_i = t_i^c - t_i^g
    float switching_cost_ms;   // c_i = S_in,i / B_pcie
    float performance_density; // k_i^b = t_i^b / s_i

    // Classification metadata
    int layer_id = -1;         // -1 if global, >= 0 for transformer layer
    bool is_moe_expert = false;
    bool is_attn = false;
    bool is_ffn = false;
};

struct atsinfer_hardware_profile {
    float pcie_bandwidth_mbps;     // B_pcie Host-to-Device in MB/s
    float pcie_d2h_bandwidth_mbps; // Device-to-Host in MB/s
    size_t gpu_vram_budget;        // M in bytes
    bool is_measured = false;      // True if dynamic CUDA profiling was executed
};

// Profile Host-to-Device transfer speed with pinned memory if CUDA is enabled
atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes);

// Profile operator latencies for model tensors
std::unordered_map<std::string, atsinfer_tensor_profile> atsinfer_profile_tensors(
    const std::vector<struct ggml_tensor *> & tensors,
    float pcie_bandwidth_mbps
);

// Cache serialization / deserialization
bool atsinfer_save_profile_cache(
    const std::string & filename,
    const atsinfer_hardware_profile & hw,
    const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles
);

bool atsinfer_load_profile_cache(
    const std::string & filename,
    atsinfer_hardware_profile & hw,
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles
);

#endif // ATSINFER_PROFILER_H
