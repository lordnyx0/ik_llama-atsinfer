#include "atsinfer/atsinfer-profiler.h"
#include "atsinfer/atsinfer-placement.h"
#include "atsinfer/atsinfer-scheduler.h"
#include "atsinfer/atsinfer-cache.h"
#include "atsinfer/atsinfer-cuda.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_tensor_cache() {
    std::cout << "[TEST] Running Tensor Cache & Eviction Test..." << std::endl;
    size_t budget = 1000 * 1024 * 1024; // 1000 MB VRAM budget
    ATSInferTensorCache cache(budget);

    cache.register_tensor("layer.0.w", 400 * 1024 * 1024, 0, false, false, ATSInferResidency::CPU_AND_GPU);
    cache.register_tensor("layer.1.w", 400 * 1024 * 1024, 1, false, false, ATSInferResidency::CPU_AND_GPU);
    cache.register_tensor("layer.2.w", 400 * 1024 * 1024, 2, false, false, ATSInferResidency::CPU_ONLY);

    cache.update_usage("layer.0.w", 100);
    cache.update_usage("layer.1.w", 200);

    std::vector<std::string> evicted;
    bool reserved = cache.reserve_gpu_space("layer.2.w", 400 * 1024 * 1024, 2, evicted);
    assert(reserved);
    assert(evicted.size() == 1);
    assert(evicted[0] == "layer.0.w"); // LRU candidate evicted

    auto * state0 = cache.get_tensor_state("layer.0.w");
    assert(state0->residency == ATSInferResidency::CPU_ONLY);
    auto * state2 = cache.get_tensor_state("layer.2.w");
    assert(state2->residency == ATSInferResidency::CPU_AND_GPU);

    std::cout << " -> Tensor Cache & Eviction Test PASSED!" << std::endl;
}

void test_cuda_manager() {
    std::cout << "[TEST] Running CUDA Manager Test..." << std::endl;
    ATSInferCudaManager cuda_mgr;
    bool inited = cuda_mgr.init(0);
    assert(inited);

    void * host_ptr = cuda_mgr.alloc_pinned_host(1024 * 1024);
    assert(host_ptr != nullptr);

    void * ev = cuda_mgr.create_event();

    bool sync_ok = cuda_mgr.wait_for_transfer_event(ev);
    assert(sync_ok);

    cuda_mgr.destroy_event(ev);
    cuda_mgr.free_pinned_host(host_ptr);
    cuda_mgr.cleanup();

    std::cout << " -> CUDA Manager Test PASSED!" << std::endl;
}

void test_profiler() {
    std::cout << "[TEST] Running Profiler Test..." << std::endl;
    auto hw = atsinfer_profile_hardware(6ULL * 1024 * 1024 * 1024); // 6GB VRAM
    assert(hw.gpu_vram_budget == 6ULL * 1024 * 1024 * 1024);
    assert(hw.pcie_bandwidth_mbps > 0.0f);

    struct ggml_tensor dummy_tensor;
    snprintf(dummy_tensor.name, sizeof(dummy_tensor.name), "blk.0.attn_q.weight");
    dummy_tensor.ne[0] = 4096;
    dummy_tensor.ne[1] = 4096;
    dummy_tensor.ne[2] = 1;
    dummy_tensor.ne[3] = 1;
    dummy_tensor.type = GGML_TYPE_F16;

    std::vector<struct ggml_tensor *> tensors = { &dummy_tensor };
    auto profiles = atsinfer_profile_tensors(tensors, hw.pcie_bandwidth_mbps);

    assert(profiles.find("blk.0.attn_q.weight") != profiles.end());
    const auto & p = profiles["blk.0.attn_q.weight"];
    assert(p.latency_reduction > 0.0f);
    assert(p.switching_cost_ms >= 0.0f);
    std::cout << " -> Profiler Test PASSED!" << std::endl;
}

void test_static_placement_dense() {
    std::cout << "[TEST] Running Static Placement Dense Test..." << std::endl;
    std::vector<atsinfer_tensor_profile> profiles;

    for (int i = 0; i < 5; ++i) {
        atsinfer_tensor_profile p;
        p.tensor_name = "layer." + std::to_string(i) + ".weight";
        p.size_bytes = 500 * 1024 * 1024; // 500 MB
        p.exec_time_cpu_ms = 20.0f;
        p.exec_time_gpu_ms = 2.0f;
        p.latency_reduction = 18.0f;
        p.switching_cost_ms = 1.0f;
        profiles.push_back(p);
    }

    size_t budget = 1200 * 1024 * 1024; // 1.2 GB VRAM (Fits 2 tensors of 500 MB)
    auto decision = atsinfer_compute_static_placement(profiles, budget, false);

    size_t gpu_count = 0;
    for (const auto & kv : decision.placement) {
        if (kv.second == ATSInferBackend::GPU) {
            gpu_count++;
        }
    }

    assert(gpu_count <= 2);
    assert(decision.total_vram_used_bytes <= budget);
    std::cout << " -> Static Placement Dense Test PASSED! (Allocated " << gpu_count << " tensors on GPU)" << std::endl;
}

void test_static_placement_moe() {
    std::cout << "[TEST] Running Static Placement MoE Test..." << std::endl;
    std::vector<atsinfer_tensor_profile> profiles;

    // Non-expert tensor
    atsinfer_tensor_profile p_attn;
    p_attn.tensor_name = "blk.0.attn_q.weight";
    p_attn.size_bytes = 200 * 1024 * 1024;
    p_attn.exec_time_cpu_ms = 15.0f;
    p_attn.exec_time_gpu_ms = 1.0f;
    p_attn.latency_reduction = 14.0f;
    p_attn.switching_cost_ms = 0.5f;
    profiles.push_back(p_attn);

    // Expert tensor
    atsinfer_tensor_profile p_exp;
    p_exp.tensor_name = "blk.0.exps.0.weight";
    p_exp.size_bytes = 400 * 1024 * 1024;
    p_exp.exec_time_cpu_ms = 30.0f;
    p_exp.exec_time_gpu_ms = 3.0f;
    p_exp.latency_reduction = 27.0f;
    p_exp.switching_cost_ms = 1.0f;
    profiles.push_back(p_exp);

    size_t budget = 500 * 1024 * 1024; // 500 MB
    auto decision = atsinfer_compute_static_placement(profiles, budget, true);

    assert(decision.placement["blk.0.attn_q.weight"] == ATSInferBackend::GPU);
    std::cout << " -> Static Placement MoE Test PASSED!" << std::endl;
}

void test_dynamic_transfer_scheduler() {
    std::cout << "[TEST] Running Dynamic Transfer Scheduler Test..." << std::endl;
    std::vector<atsinfer_tensor_profile> profiles;

    atsinfer_tensor_profile p1;
    p1.tensor_name = "layer.0.weight";
    p1.size_bytes = 100 * 1024 * 1024;
    p1.exec_time_cpu_ms = 25.0f;
    p1.exec_time_gpu_ms = 2.0f;
    p1.latency_reduction = 23.0f;
    p1.switching_cost_ms = 2.0f;
    profiles.push_back(p1);

    std::unordered_map<std::string, ATSInferBackend> static_map;
    static_map["layer.0.weight"] = ATSInferBackend::CPU;

    auto dyn_sched = atsinfer_dynamic_transfer_schedule(profiles, static_map, 16000.0f);
    assert(dyn_sched.promoted_tensors_to_gpu.size() == 1);
    assert(dyn_sched.promoted_tensors_to_gpu[0] == "layer.0.weight");
    std::cout << " -> Dynamic Transfer Scheduler Test PASSED!" << std::endl;
}

void test_load_aware_rescheduler() {
    std::cout << "[TEST] Running Load-Aware Rescheduler Test..." << std::endl;
    ATSInferRescheduler rescheduler(0.15f, 5);

    // Initial state
    assert(rescheduler.should_reschedule(0.0f, 40.0f, 40.0f) == true);
    rescheduler.record_reschedule_event(40.0f);

    // Small deviation (5%) - should NOT reschedule
    assert(rescheduler.should_reschedule(40.0f, 42.0f, 40.0f) == false);

    // Large deviation (25%) but not enough time elapsed (< 5 * TPOT)
    assert(rescheduler.should_reschedule(40.0f, 50.0f, 40.0f) == false);

    // Advance time beyond 5 * TPOT
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);

    // Large deviation now - SHOULD reschedule
    assert(rescheduler.should_reschedule(40.0f, 52.0f, 40.0f) == true);
    std::cout << " -> Load-Aware Rescheduler Test PASSED!" << std::endl;
}

void test_profile_serialization() {
    std::cout << "[TEST] Running Profile Serialization Test..." << std::endl;
    atsinfer_hardware_profile hw_orig;
    hw_orig.pcie_bandwidth_mbps = 24000.0f;
    hw_orig.pcie_d2h_bandwidth_mbps = 22000.0f;
    hw_orig.gpu_vram_budget = 8192ULL * 1024 * 1024;
    hw_orig.is_measured = true;

    std::unordered_map<std::string, atsinfer_tensor_profile> profiles_orig;
    atsinfer_tensor_profile p;
    p.tensor_name = "blk.3.attn_q.weight";
    p.size_bytes = 100 * 1024 * 1024;
    p.exec_time_cpu_ms = 12.5f;
    p.exec_time_gpu_ms = 1.2f;
    p.layer_id = 3;
    p.is_attn = true;
    profiles_orig[p.tensor_name] = p;

    std::string cache_path = "test_atsinfer_cache.txt";
    bool saved = atsinfer_save_profile_cache(cache_path, hw_orig, profiles_orig);
    assert(saved);

    atsinfer_hardware_profile hw_loaded;
    std::unordered_map<std::string, atsinfer_tensor_profile> profiles_loaded;
    bool loaded = atsinfer_load_profile_cache(cache_path, hw_loaded, profiles_loaded);
    assert(loaded);

    assert(hw_loaded.pcie_bandwidth_mbps == hw_orig.pcie_bandwidth_mbps);
    assert(hw_loaded.gpu_vram_budget == hw_orig.gpu_vram_budget);
    assert(profiles_loaded.find("blk.3.attn_q.weight") != profiles_loaded.end());
    assert(profiles_loaded["blk.3.attn_q.weight"].layer_id == 3);
    assert(profiles_loaded["blk.3.attn_q.weight"].is_attn == true);

    std::remove(cache_path.c_str());
    std::cout << " -> Profile Serialization Test PASSED!" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "    ATSInfer Unit Test Suite Execution    " << std::endl;
    std::cout << "==========================================" << std::endl;

    test_profiler();
    test_static_placement_dense();
    test_static_placement_moe();
    test_dynamic_transfer_scheduler();
    test_load_aware_rescheduler();
    test_profile_serialization();
    test_tensor_cache();
    test_cuda_manager();

    std::cout << "==========================================" << std::endl;
    std::cout << "   ALL ATSINFER UNIT TESTS PASSED (8/8)   " << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}
