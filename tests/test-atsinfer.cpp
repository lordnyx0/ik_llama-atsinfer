#include "atsinfer/atsinfer-profiler.h"
#include "atsinfer/atsinfer-placement.h"
#include "atsinfer/atsinfer-scheduler.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

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

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "    ATSInfer Unit Test Suite Execution    " << std::endl;
    std::cout << "==========================================" << std::endl;

    test_profiler();
    test_static_placement_dense();
    test_static_placement_moe();
    test_dynamic_transfer_scheduler();
    test_load_aware_rescheduler();

    std::cout << "==========================================" << std::endl;
    std::cout << "   ALL ATSINFER UNIT TESTS PASSED (5/5)   " << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}
