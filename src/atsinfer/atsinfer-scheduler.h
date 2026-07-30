#ifndef ATSINFER_SCHEDULER_H
#define ATSINFER_SCHEDULER_H

#include "atsinfer-placement.h"
#include <vector>
#include <string>
#include <unordered_map>

struct atsinfer_dynamic_schedule {
    std::vector<std::string> promoted_tensors_to_gpu;
    float estimated_round_latency_ms;
};

class ATSInferRescheduler {
public:
    ATSInferRescheduler(float deviation_threshold = 0.15f, int min_tpot_multiplier = 5);

    // Checks whether load deviation requires triggering dynamic re-scheduling (Algorithm 3)
    bool should_reschedule(float last_measured_latency_ms, float current_measured_latency_ms, float current_tpot_ms);

    void record_reschedule_event(float current_tpot_ms);

private:
    float threshold;
    int tpot_multiplier;
    float last_reschedule_time_ms;
    float accumulated_time_ms;
};

// Algoritmo 2: Dynamic Transfer Scheduling
atsinfer_dynamic_schedule atsinfer_dynamic_transfer_schedule(
    const std::vector<atsinfer_tensor_profile> & tensors,
    const std::unordered_map<std::string, ATSInferBackend> & static_placement,
    float pcie_bandwidth_mbps
);

#endif // ATSINFER_SCHEDULER_H
