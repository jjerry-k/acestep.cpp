#pragma once
// apg-core.h: APG (Adaptive Projected Guidance) primitives
//
// Extracted from dit-sampler.h so guidance modes can share the core functions.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include <cmath>
#include <cstring>
#include <vector>

// Momentum buffer for APG running average smoothing.
// Matches Python APGMomentumBuffer with momentum=-0.75
struct APGMomentumBuffer {
    double              momentum;
    std::vector<double> running_average;
    bool                initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++) {
                running_average[i] = values[i] + momentum * running_average[i];
            }
        }
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(const double * v0, const double * v1,
                         double * out_par, double * out_orth,
                         int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++) {
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        }
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++) {
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);
        }

        for (int t = 0; t < T; t++) {
            int    idx    = t * Oc + c;
            double v1n    = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward() exactly:
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
// Internal computation in double precision (Python uses .double()).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold = 2.5f) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    std::vector<double> diff(n);
    for (int i = 0; i < n; i++) {
        diff[i] = (double) pred_cond[i] - (double) pred_uncond[i];
    }

    // 2. momentum update, then use smoothed diff
    mbuf.update(diff.data(), n);
    memcpy(diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++) {
                norm2 += diff[t * Oc + c] * diff[t * Oc + c];
            }
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s    = (norm > 1e-60) ? fmin(1.0, (double) norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++) {
                    diff[t * Oc + c] *= s;
                }
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    std::vector<double> pred_cond_d(n), par(n), orth(n);
    for (int i = 0; i < n; i++) {
        pred_cond_d[i] = (double) pred_cond[i];
    }
    apg_project(diff.data(), pred_cond_d.data(), par.data(), orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double) guidance_scale - 1.0;
    for (int i = 0; i < n; i++) {
        result[i] = (float) ((double) pred_cond[i] + w * orth[i]);
    }
}
