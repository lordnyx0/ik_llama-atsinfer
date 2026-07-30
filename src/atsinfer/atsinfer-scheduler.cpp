#include "atsinfer-scheduler.h"
#include <algorithm>
#include <cmath>

ATSInferRescheduler::ATSInferRescheduler(float deviation_threshold, int min_tpot_multiplier)
    : threshold(deviation_threshold), tpot_multiplier(min_tpot_multiplier), last_reschedule_time_ms(0.0f), accumulated_time_ms(0.0f) {}

bool ATSInferRescheduler::should_reschedule(
    float last_measured_latency_ms,
    float current_measured_latency_ms,
    float current_tpot_ms) {
    
    if (last_measured_latency_ms <= 0.0f) return true;

    float deviation = std::abs(current_measured_latency_ms - last_measured_latency_ms) / last_measured_latency_ms;
    float min_interval = current_tpot_ms * (float)tpot_multiplier;

    accumulated_time_ms += current_tpot_ms;

    if (deviation >= threshold && (accumulated_time_ms - last_reschedule_time_ms) >= min_interval) {
        return true;
    }

    return false;
}

void ATSInferRescheduler::record_reschedule_event(float current_tpot_ms) {
    last_reschedule_time_ms = accumulated_time_ms;
}

atsinfer_dynamic_schedule atsinfer_dynamic_transfer_schedule(
    const std::vector<atsinfer_tensor_profile> & tensors,
    const std::unordered_map<std::string, ATSInferBackend> & static_placement,
    float pcie_bandwidth_mbps) {
    
    atsinfer_dynamic_schedule schedule;
    size_t n = tensors.size();
    if (n == 0) return schedule;

    if (pcie_bandwidth_mbps <= 0.0f) pcie_bandwidth_mbps = 16000.0f;

    // Calculate interval overlap seg(j, i) and exposed transfer cost delta_i
    float accumulated_cpu_comp_time = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        const auto & t = tensors[i];
        auto it = static_placement.find(t.tensor_name);
        ATSInferBackend default_b = (it != static_placement.end()) ? it->second : ATSInferBackend::CPU;

        if (default_b == ATSInferBackend::CPU) {
            float weight_transfer_time = t.switching_cost_ms;
            float overlap_window = accumulated_cpu_comp_time;
            float exposed_transfer_cost = std::max(0.0f, weight_transfer_time - overlap_window);

            // If promoting this CPU tensor to GPU yields net latency gain:
            if (t.latency_reduction > exposed_transfer_cost) {
                schedule.promoted_tensors_to_gpu.push_back(t.tensor_name);
            }
            accumulated_cpu_comp_time += t.exec_time_cpu_ms;
        }
    }

    return schedule;
}
