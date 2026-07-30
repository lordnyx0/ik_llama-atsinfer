#include "atsinfer-profiler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

static float measure_cuda_h2d_bandwidth(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, size_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || !host_ptr) return 0.0f;

    err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess || !device_ptr) {
        cudaFreeHost(host_ptr);
        return 0.0f;
    }

    memset(host_ptr, 0xAB, size_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemcpyAsync(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, 0);
    cudaDeviceSynchronize();

    // Measure H2D
    cudaEventRecord(start, 0);
    for (int i = 0; i < 5; ++i) {
        cudaMemcpyAsync(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_ptr);
    cudaFreeHost(host_ptr);

    if (elapsed_ms <= 0.0001f) return 0.0f;
    float total_mb = (float)(size_bytes * 5) / (1024.0f * 1024.0f);
    return (total_mb / (elapsed_ms / 1000.0f)); // MB/s
#else
    (void)size_bytes;
    return 0.0f;
#endif
}

static float measure_cuda_d2h_bandwidth(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, size_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || !host_ptr) return 0.0f;

    err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess || !device_ptr) {
        cudaFreeHost(host_ptr);
        return 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemcpyAsync(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();

    // Measure D2H
    cudaEventRecord(start, 0);
    for (int i = 0; i < 5; ++i) {
        cudaMemcpyAsync(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_ptr);
    cudaFreeHost(host_ptr);

    if (elapsed_ms <= 0.0001f) return 0.0f;
    float total_mb = (float)(size_bytes * 5) / (1024.0f * 1024.0f);
    return (total_mb / (elapsed_ms / 1000.0f)); // MB/s
#else
    (void)size_bytes;
    return 0.0f;
#endif
}

atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes) {
    atsinfer_hardware_profile hw;
    hw.gpu_vram_budget = vram_budget_bytes;
    hw.pcie_bandwidth_mbps = 16000.0f;     // Default fallback (PCIe 4.0 x16 conservative)
    hw.pcie_d2h_bandwidth_mbps = 14000.0f; // Default fallback
    hw.is_measured = false;

    // Measure H2D and D2H bandwidth for 64MB transfer size if CUDA is enabled
    float h2d = measure_cuda_h2d_bandwidth(64 * 1024 * 1024);
    float d2h = measure_cuda_d2h_bandwidth(64 * 1024 * 1024);

    if (h2d > 100.0f) {
        hw.pcie_bandwidth_mbps = h2d;
        hw.is_measured = true;
    }
    if (d2h > 100.0f) {
        hw.pcie_d2h_bandwidth_mbps = d2h;
    }

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

        // Extract layer index if named like "blk.X." or "layers.X."
        p.layer_id = -1;
        std::string name_str(tensor->name);
        if (name_str.find("blk.") == 0 || name_str.find("layers.") == 0) {
            size_t dot1 = name_str.find('.');
            if (dot1 != std::string::npos) {
                size_t dot2 = name_str.find('.', dot1 + 1);
                if (dot2 != std::string::npos) {
                    try {
                        p.layer_id = std::stoi(name_str.substr(dot1 + 1, dot2 - dot1 - 1));
                    } catch (...) {
                        p.layer_id = -1;
                    }
                }
            }
        }

        // Tensor role classification
        p.is_moe_expert = (name_str.find("ffn_exp") != std::string::npos ||
                           name_str.find("exps") != std::string::npos ||
                           name_str.find("experts") != std::string::npos);
        p.is_attn = (name_str.find("attn") != std::string::npos ||
                     name_str.find("self_attn") != std::string::npos);
        p.is_ffn = (name_str.find("ffn") != std::string::npos ||
                    name_str.find("mlp") != std::string::npos);

        // Empirical estimation derived from operator type and tensor dimensions
        float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);
        
        // Base CPU vs GPU throughput heuristic per MB
        // GPU is ~5x-10x faster on matrix multiplications (weight tensors)
        p.exec_time_cpu_ms = std::max(0.01f, size_mb * 0.45f);
        p.exec_time_gpu_ms = std::max(0.002f, size_mb * 0.06f);
        
        p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;
        p.switching_cost_ms = (size_mb) / (pcie_bandwidth_mbps / 1000.0f); // Transfer time in ms
        p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);

        profiles[p.tensor_name] = p;
    }

    return profiles;
}

bool atsinfer_save_profile_cache(
    const std::string & filename,
    const atsinfer_hardware_profile & hw,
    const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles) {
    
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "# ATSInfer Profile Cache\n";
    out << "HW " << hw.pcie_bandwidth_mbps << " " << hw.pcie_d2h_bandwidth_mbps << " "
        << hw.gpu_vram_budget << " " << (hw.is_measured ? 1 : 0) << "\n";

    for (const auto & kv : profiles) {
        const auto & p = kv.second;
        out << "TENSOR " << p.tensor_name << " " << p.size_bytes << " "
            << p.exec_time_cpu_ms << " " << p.exec_time_gpu_ms << " "
            << p.layer_id << " " << (p.is_moe_expert ? 1 : 0) << " "
            << (p.is_attn ? 1 : 0) << " " << (p.is_ffn ? 1 : 0) << "\n";
    }

    return true;
}

bool atsinfer_load_profile_cache(
    const std::string & filename,
    atsinfer_hardware_profile & hw,
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles) {

    std::ifstream in(filename);
    if (!in.is_open()) return false;

    profiles.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "HW") {
            int is_meas = 0;
            iss >> hw.pcie_bandwidth_mbps >> hw.pcie_d2h_bandwidth_mbps >> hw.gpu_vram_budget >> is_meas;
            hw.is_measured = (is_meas != 0);
        } else if (tag == "TENSOR") {
            atsinfer_tensor_profile p;
            int is_moe = 0, is_attn = 0, is_ffn = 0;
            iss >> p.tensor_name >> p.size_bytes >> p.exec_time_cpu_ms >> p.exec_time_gpu_ms
                >> p.layer_id >> is_moe >> is_attn >> is_ffn;

            p.is_moe_expert = (is_moe != 0);
            p.is_attn = (is_attn != 0);
            p.is_ffn = (is_ffn != 0);

            p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;
            float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);
            float bw = hw.pcie_bandwidth_mbps > 0.0f ? hw.pcie_bandwidth_mbps : 16000.0f;
            p.switching_cost_ms = (size_mb) / (bw / 1000.0f);
            p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);

            profiles[p.tensor_name] = p;
        }
    }

    return true;
}
