#include "atsinfer-profiler.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>

atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes) {
    atsinfer_hardware_profile hw;
    hw.gpu_vram_budget = vram_budget_bytes;
    
    // Default PCIe 4.0 x16 conservative bandwidth estimate ~16000 MB/s if not measured
    hw.pcie_bandwidth_mbps = 16000.0f;
    return hw;
}

std::unordered_map<std::string, atsinfer_tensor_profile> atsinfer_profile_tensors(
    const std::vector<struct ggml_tensor *> & tensors,
    float pcie_bandwidth_mbps) {
    
    std::unordered_map<std::string, atsinfer_tensor_profile> profiles;
    if (pcie_bandwidth_mbps <= 0.0f) {
        pcie_bandwidth_mbps = 16000.0f;
    }

    for (const auto * tensor : tensors) {
        if (!tensor || !tensor->name[0]) continue;

        atsinfer_tensor_profile p;
        p.tensor_name = tensor->name;
        p.size_bytes = ggml_nbytes(tensor);

        // Empirical estimation derived from operator type and tensor dimensions
        float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);
        
        // Base CPU vs GPU throughput heuristic per MB
        // GPU is ~4x-10x faster on matrix multiplications (weight tensors)
        p.exec_time_cpu_ms = std::max(0.01f, size_mb * 0.45f);
        p.exec_time_gpu_ms = std::max(0.002f, size_mb * 0.06f);
        
        p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;
        p.switching_cost_ms = (size_mb) / (pcie_bandwidth_mbps / 1000.0f); // Transfer time in ms
        p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);

        profiles[p.tensor_name] = p;
    }

    return profiles;
}
