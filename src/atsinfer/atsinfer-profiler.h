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
};

struct atsinfer_hardware_profile {
    float pcie_bandwidth_mbps; // B_pcie in MB/s
    size_t gpu_vram_budget;    // M in bytes
};

// Profile Host-to-Device transfer speed
atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes);

// Profile operator latencies for model tensors
std::unordered_map<std::string, atsinfer_tensor_profile> atsinfer_profile_tensors(
    const std::vector<struct ggml_tensor *> & tensors,
    float pcie_bandwidth_mbps
);

#endif // ATSINFER_PROFILER_H
