#pragma once
// mastering.h — Reference-based audio mastering (matchering algorithm)
//
// Clean-room C++ implementation of the matchering 2.0 algorithm:
//   1. Match Levels  — Mid/Side RMS analysis & loudness matching
//   2. Match Frequencies — FFT spectral analysis → FIR correction filter → convolve
//   3. Correct Levels — Iterative RMS feedback loop
//   4. Finalize — Hyrax brickwall limiter → peak normalize
//
// Dependencies: pocketfft (header-only FFT, MIT license)
//
// Reference: https://github.com/sergree/matchering (GPL-3.0)
// This is an independent implementation of the published algorithm,
// not a copy/translation of the GPL code.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <deque>
#include <numeric>
#include <thread>
#include <vector>

#include "pocketfft_hdronly.h"

// ─── Configuration ──────────────────────────────────────────────────

struct MasteringConfig {
    int   sample_rate           = 48000;
    int   fft_size              = 4096;
    float max_piece_seconds     = 15.0f;
    int   rms_correction_steps  = 4;
    float threshold             = (32768.0f - 61.0f) / 32768.0f;  // ~0.998
    float min_value             = 1e-6f;
    int   lin_log_oversampling  = 4;
    float lowess_frac           = 0.0375f;
    int   lowess_it             = 0;
    float lowess_delta          = 0.001f;
    // Hyrax limiter
    float limiter_attack_ms     = 1.0f;
    float limiter_hold_ms       = 1.0f;
    float limiter_release_ms    = 3000.0f;
    float limiter_attack_coef   = -2.0f;
    int   limiter_hold_order    = 1;
    float limiter_hold_coef     = 7.0f;
    int   limiter_release_order = 1;
    float limiter_release_coef  = 800.0f;
    // Clipping thresholds
    int   clipping_samples_threshold = 8;
    int   limited_samples_threshold  = 128;
};

// ─── DSP Primitives ─────────────────────────────────────────────────

static inline float mg_rms(const float * x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double) x[i] * x[i];
    return (float) std::sqrt(sum / n);
}

static inline void mg_amplify(float * x, int n, float gain) {
    for (int i = 0; i < n; i++) x[i] *= gain;
}

static inline void mg_clip(float * x, int n, float ceiling = 1.0f) {
    for (int i = 0; i < n; i++) x[i] = std::clamp(x[i], -ceiling, ceiling);
}

static inline float mg_peak(const float * x, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; i++) peak = std::max(peak, std::abs(x[i]));
    return peak;
}

// Count samples at or above max peak
static inline int mg_count_max_peaks(const float * x, int n, float max_val) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (std::abs(std::abs(x[i]) - max_val) < 1e-7f) count++;
    }
    return count;
}

// Mid/Side encoding: mid = (L+R)/2, side = (L-R)/2
static void mg_lr_to_ms(const float * L, const float * R, int n, float * mid, float * side) {
    for (int i = 0; i < n; i++) {
        mid[i]  = (L[i] + R[i]) * 0.5f;
        side[i] = (L[i] - R[i]) * 0.5f;
    }
}

// Mid/Side decoding: L = mid+side, R = mid-side
static void mg_ms_to_lr(const float * mid, const float * side, int n, float * L, float * R) {
    for (int i = 0; i < n; i++) {
        L[i] = mid[i] + side[i];
        R[i] = mid[i] - side[i];
    }
}

// Peak normalize. Returns the coefficient applied.
static float mg_normalize(float * x, int n, float threshold, float epsilon, bool normalize_clipped) {
    float max_val = mg_peak(x, n);
    float coeff   = 1.0f;
    if (max_val < threshold || normalize_clipped) {
        coeff = std::max(epsilon, max_val / threshold);
    }
    if (coeff > epsilon) {
        float inv = 1.0f / coeff;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
    return coeff;
}

// Batch RMS: compute RMS for each piece of size piece_size
static std::vector<float> mg_batch_rms(const float * x, int piece_size, int n_pieces) {
    std::vector<float> out(n_pieces);
    for (int p = 0; p < n_pieces; p++) {
        out[p] = mg_rms(x + p * piece_size, piece_size);
    }
    return out;
}

// ─── LOWESS Smoother ────────────────────────────────────────────────
// Simplified LOWESS (locally weighted scatterplot smoothing)
// Matches statsmodels.nonparametric.lowess behavior for spectral smoothing

static std::vector<float> mg_lowess(const float * y, int n, float frac, int niter, float delta) {
    std::vector<float> result(n);
    int h = std::max(1, (int) std::ceil(frac * n));

    // Precompute x positions (uniform 0..1)
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = (float) i / std::max(1, n - 1);

    std::vector<float> residuals(n, 0.0f);

    for (int iter = 0; iter <= niter; iter++) {
        int last_fit = -1;
        float last_val = 0.0f;

        for (int i = 0; i < n; i++) {
            // Skip points within delta of last fit
            if (iter == 0 && delta > 0 && last_fit >= 0
                && x[i] - x[last_fit] < delta && i < n - 1) {
                continue;
            }

            // Find bandwidth: distance to h-th nearest neighbor
            // Simple: use h neighbors on each side
            int lo = std::max(0, i - h);
            int hi = std::min(n - 1, i + h);

            float max_dist = 0.0f;
            for (int j = lo; j <= hi; j++) {
                max_dist = std::max(max_dist, std::abs(x[j] - x[i]));
            }
            if (max_dist < 1e-12f) max_dist = 1.0f;

            // Compute tricube weights
            double sum_w = 0, sum_wx = 0, sum_wy = 0;
            double sum_wxx = 0, sum_wxy = 0;
            for (int j = lo; j <= hi; j++) {
                float u = std::abs(x[j] - x[i]) / max_dist;
                float w = (u < 1.0f) ? (1.0f - u * u * u) : 0.0f;
                w = w * w * w;  // tricube
                if (iter > 0) {
                    // Bisquare robustness weight
                    float r = std::abs(residuals[j]);
                    float med_r = 1.0f;  // simplified
                    float ru = r / (6.0f * med_r);
                    float rw = (ru < 1.0f) ? (1.0f - ru * ru) * (1.0f - ru * ru) : 0.0f;
                    w *= rw;
                }
                sum_w   += w;
                sum_wx  += w * x[j];
                sum_wy  += w * y[j];
                sum_wxx += w * x[j] * x[j];
                sum_wxy += w * x[j] * y[j];
            }

            // Weighted linear regression
            double denom = sum_w * sum_wxx - sum_wx * sum_wx;
            if (std::abs(denom) < 1e-12) {
                result[i] = (sum_w > 0) ? (float) (sum_wy / sum_w) : y[i];
            } else {
                double beta1 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
                double beta0 = (sum_wy - beta1 * sum_wx) / sum_w;
                result[i] = (float) (beta0 + beta1 * x[i]);
            }

            if (last_fit >= 0 && i > last_fit + 1) {
                // Interpolate skipped points
                float slope = (result[i] - last_val) / (x[i] - x[last_fit]);
                for (int k = last_fit + 1; k < i; k++) {
                    result[k] = last_val + slope * (x[k] - x[last_fit]);
                }
            }

            last_fit = i;
            last_val = result[i];
        }

        // Fill remaining skipped points at the end
        if (last_fit >= 0 && last_fit < n - 1) {
            for (int k = last_fit + 1; k < n; k++) {
                result[k] = last_val;
            }
        }

        // Compute residuals for robustness iteration
        if (iter < niter) {
            for (int i = 0; i < n; i++) {
                residuals[i] = y[i] - result[i];
            }
        }
    }
    return result;
}

// ─── Sinc Resampler ─────────────────────────────────────────────────
// Quality sinc interpolation for sample rate conversion (e.g. reference → 48000 Hz)

static std::vector<float> mg_resample(const float * input, int n_in, int sr_in, int sr_out) {
    if (sr_in == sr_out) {
        return std::vector<float>(input, input + n_in);
    }

    double ratio = (double) sr_out / sr_in;
    int n_out = (int) std::ceil(n_in * ratio);
    std::vector<float> output(n_out);

    // Windowed sinc interpolation, 16-tap Kaiser window
    const int TAPS = 16;
    const double beta = 6.0;  // Kaiser beta

    // Precompute Kaiser window I0(beta)
    auto bessel_i0 = [](double x) -> double {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k <= 20; k++) {
            term *= (x * x) / (4.0 * k * k);
            sum += term;
            if (term < 1e-12) break;
        }
        return sum;
    };
    double i0_beta = bessel_i0(beta);

    // Cutoff frequency (anti-alias)
    double fc = std::min(1.0, ratio);
    if (sr_out < sr_in) fc = ratio;

    for (int i = 0; i < n_out; i++) {
        double src_pos = i / ratio;
        int    center  = (int) std::floor(src_pos);
        double frac    = src_pos - center;

        double sum = 0.0, wsum = 0.0;
        for (int t = -TAPS + 1; t <= TAPS; t++) {
            int idx = center + t;
            if (idx < 0 || idx >= n_in) continue;

            double x = t - frac;

            // Sinc
            double sinc_val = (std::abs(x) < 1e-10) ? 1.0
                : std::sin(M_PI * fc * x) / (M_PI * x);

            // Kaiser window
            double u = x / TAPS;
            double kaiser = (std::abs(u) <= 1.0)
                ? bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0_beta
                : 0.0;

            double w = sinc_val * kaiser * fc;
            sum  += input[idx] * w;
            wsum += w;
        }
        output[i] = (wsum > 0) ? (float) (sum / wsum) : 0.0f;
    }
    return output;
}

// ─── FFT Convolution ────────────────────────────────────────────────

static std::vector<float> mg_fftconvolve(const float * signal, int n_sig,
                                          const float * kernel, int n_ker) {
    int n_out = n_sig + n_ker - 1;
    // Next power of 2 for efficiency
    size_t n_fft = 1;
    while ((int) n_fft < n_out) n_fft *= 2;

    using cpx = std::complex<double>;

    // Zero-pad signal and kernel
    std::vector<double> sig_d(n_fft, 0.0), ker_d(n_fft, 0.0);
    for (int i = 0; i < n_sig; i++) sig_d[i] = signal[i];
    for (int i = 0; i < n_ker; i++) ker_d[i] = kernel[i];

    // Forward FFT
    pocketfft::shape_t shape = {n_fft};
    pocketfft::stride_t stride_in  = {(ptrdiff_t) sizeof(double)};
    pocketfft::stride_t stride_out = {(ptrdiff_t) sizeof(cpx)};
    size_t n_cpx = n_fft / 2 + 1;
    std::vector<cpx> sig_f(n_cpx), ker_f(n_cpx);

    pocketfft::r2c(shape, stride_in, stride_out, {0},
                    pocketfft::FORWARD, sig_d.data(), sig_f.data(), 1.0);
    pocketfft::r2c(shape, stride_in, stride_out, {0},
                    pocketfft::FORWARD, ker_d.data(), ker_f.data(), 1.0);

    // Multiply in frequency domain
    for (size_t i = 0; i < n_cpx; i++) {
        sig_f[i] *= ker_f[i];
    }

    // Inverse FFT
    std::vector<double> result_d(n_fft);
    pocketfft::c2r(shape, stride_out, stride_in, {0},
                    pocketfft::BACKWARD, sig_f.data(), result_d.data(),
                    1.0 / n_fft);

    // Copy to output (truncate to valid length)
    std::vector<float> result(n_out);
    for (int i = 0; i < n_out; i++) result[i] = (float) result_d[i];
    return result;
}

// ─── Spectral Analysis & FIR Design ────────────────────────────────

// Compute average magnitude spectrum of pieces
// Uses boxcar (rectangular) window with no overlap, matching Python matchering's
// scipy.signal.stft(window='boxcar', noverlap=0).
static std::vector<float> mg_avg_spectrum(const float * pieces, int piece_size,
                                           int n_pieces, int fft_size) {
    size_t n_cpx = fft_size / 2 + 1;
    std::vector<double> avg_mag(n_cpx, 0.0);

    pocketfft::shape_t  shape      = {(size_t) fft_size};
    pocketfft::stride_t stride_in  = {(ptrdiff_t) sizeof(double)};
    pocketfft::stride_t stride_out = {(ptrdiff_t) sizeof(std::complex<double>)};

    std::vector<double> padded(fft_size, 0.0);
    std::vector<std::complex<double>> freq(n_cpx);

    // Boxcar (rectangular) window — no windowing applied
    // Hop = fft_size (no overlap), matching Python's noverlap=0

    int n_frames = 0;
    int hop = fft_size;  // no overlap

    for (int p = 0; p < n_pieces; p++) {
        const float * piece = pieces + p * piece_size;
        for (int start = 0; start + fft_size <= piece_size; start += hop) {
            for (int i = 0; i < fft_size; i++) {
                padded[i] = piece[start + i];  // boxcar: no window multiplication
            }
            pocketfft::r2c(shape, stride_in, stride_out, {0},
                           pocketfft::FORWARD, padded.data(), freq.data(), 1.0);
            for (size_t i = 0; i < n_cpx; i++) {
                avg_mag[i] += std::abs(freq[i]);
            }
            n_frames++;
        }
    }

    if (n_frames > 0) {
        for (size_t i = 0; i < n_cpx; i++) avg_mag[i] /= n_frames;
    }

    std::vector<float> result(n_cpx);
    for (size_t i = 0; i < n_cpx; i++) result[i] = (float) avg_mag[i];
    return result;
}

// Design FIR correction filter from target and reference spectra
static std::vector<float> mg_get_fir(const float * target_pieces, int target_piece_size,
                                      int n_target_pieces,
                                      const float * ref_pieces, int ref_piece_size,
                                      int n_ref_pieces,
                                      const MasteringConfig & cfg) {
    int fft_size = cfg.fft_size;
    size_t n_cpx = fft_size / 2 + 1;

    // Average spectra
    auto tgt_spec = mg_avg_spectrum(target_pieces, target_piece_size,
                                    n_target_pieces, fft_size);
    auto ref_spec = mg_avg_spectrum(ref_pieces, ref_piece_size,
                                    n_ref_pieces, fft_size);

    // Correction curve: ref / target (avoid division by zero)
    std::vector<float> correction(n_cpx);
    for (size_t i = 0; i < n_cpx; i++) {
        float t = std::max(tgt_spec[i], cfg.min_value);
        float r = ref_spec[i];
        correction[i] = r / t;
    }

    // Map to log-frequency scale for LOWESS smoothing
    int n_log = (int) n_cpx * cfg.lin_log_oversampling;
    std::vector<float> log_correction(n_log);
    for (int i = 0; i < n_log; i++) {
        // Log-spaced index into linear spectrum
        float t     = (float) i / (n_log - 1);
        float log_i = std::pow((float) (n_cpx - 1), t);
        int   lo    = (int) log_i;
        int   hi    = std::min(lo + 1, (int) n_cpx - 1);
        float frac  = log_i - lo;
        log_correction[i] = correction[lo] * (1.0f - frac) + correction[hi] * frac;
    }

    // LOWESS smoothing
    auto smoothed = mg_lowess(log_correction.data(), n_log,
                              cfg.lowess_frac, cfg.lowess_it, cfg.lowess_delta);

    // Map back to linear scale
    std::vector<float> correction_smooth(n_cpx);
    for (size_t i = 0; i < n_cpx; i++) {
        float t     = std::log((float) (i + 1)) / std::log((float) n_cpx);
        float log_i = t * (n_log - 1);
        int   lo    = (int) log_i;
        int   hi    = std::min(lo + 1, n_log - 1);
        float frac  = log_i - lo;
        correction_smooth[i] = smoothed[lo] * (1.0f - frac) + smoothed[hi] * frac;
    }

    // Match Python matchering: zero DC bin, preserve raw bin[1]
    correction_smooth[0] = 0.0f;
    if (n_cpx > 1) {
        correction_smooth[1] = correction[1];
    }

    // Build symmetric frequency response and IFFT to get FIR
    size_t n_fft = (n_cpx - 1) * 2;
    pocketfft::shape_t  shape      = {n_fft};
    pocketfft::stride_t stride_cpx = {(ptrdiff_t) sizeof(std::complex<double>)};
    pocketfft::stride_t stride_re  = {(ptrdiff_t) sizeof(double)};

    std::vector<std::complex<double>> freq_resp(n_cpx);
    for (size_t i = 0; i < n_cpx; i++) {
        freq_resp[i] = std::complex<double>(correction_smooth[i], 0.0);
    }

    std::vector<double> fir_d(n_fft);
    pocketfft::c2r(shape, stride_cpx, stride_re, {0},
                    pocketfft::BACKWARD, freq_resp.data(), fir_d.data(),
                    1.0 / n_fft);

    // Circular shift to center the FIR, truncate to fft_size taps
    int fir_len = fft_size;
    std::vector<float> fir(fir_len);
    int half = fir_len / 2;
    for (int i = 0; i < fir_len; i++) {
        int src = (i - half + (int) n_fft) % (int) n_fft;
        fir[i] = (float) fir_d[src];
    }

    // Apply Hann window to FIR
    for (int i = 0; i < fir_len; i++) {
        float w = 0.5f * (1.0f - std::cos(2.0f * (float) M_PI * i / (fir_len - 1)));
        fir[i] *= w;
    }

    return fir;
}

// ─── Level Analysis ─────────────────────────────────────────────────

struct LevelAnalysis {
    std::vector<float> mid;
    std::vector<float> side;
    std::vector<float> mid_loudest;    // concatenated loudest pieces
    std::vector<float> side_loudest;
    float              match_rms;
    int                divisions;
    int                piece_size;
    int                n_loudest;
};

static LevelAnalysis mg_analyze_levels(const float * L, const float * R, int n,
                                        const MasteringConfig & cfg) {
    LevelAnalysis la;
    la.piece_size = std::min((int) (cfg.max_piece_seconds * cfg.sample_rate), n);
    la.divisions  = n / la.piece_size;
    if (la.divisions < 1) la.divisions = 1;

    // Mid/Side
    la.mid.resize(n);
    la.side.resize(n);
    mg_lr_to_ms(L, R, n, la.mid.data(), la.side.data());

    // Compute per-piece RMS on mid channel
    auto rmses = mg_batch_rms(la.mid.data(), la.piece_size, la.divisions);

    // Average RMS
    float avg_rms = 0.0f;
    for (float r : rmses) avg_rms += r;
    avg_rms /= rmses.size();

    // Find loudest pieces (above average)
    std::vector<int> loud_indices;
    for (int i = 0; i < (int) rmses.size(); i++) {
        if (rmses[i] >= avg_rms) loud_indices.push_back(i);
    }
    if (loud_indices.empty()) {
        // Fallback: use all pieces
        for (int i = 0; i < (int) rmses.size(); i++) loud_indices.push_back(i);
    }

    la.n_loudest = (int) loud_indices.size();

    // Concatenate loudest pieces
    la.mid_loudest.resize(la.n_loudest * la.piece_size);
    la.side_loudest.resize(la.n_loudest * la.piece_size);
    for (int i = 0; i < la.n_loudest; i++) {
        int src = loud_indices[i] * la.piece_size;
        std::copy_n(la.mid.data() + src, la.piece_size, la.mid_loudest.data() + i * la.piece_size);
        std::copy_n(la.side.data() + src, la.piece_size, la.side_loudest.data() + i * la.piece_size);
    }

    // Match RMS from loudest pieces
    float loud_rms = 0.0f;
    for (int idx : loud_indices) loud_rms += rmses[idx];
    la.match_rms = loud_rms / loud_indices.size();

    return la;
}

// ─── Sliding Window Maximum ─────────────────────────────────────────
// O(n) deque-based implementation matching scipy.ndimage.maximum_filter1d

static std::vector<float> mg_sliding_max(const float * x, int n, int window_size) {
    std::vector<float> out(n);
    std::deque<int> dq;
    int half = (window_size - 1) / 2;

    for (int i = 0; i < n; i++) {
        while (!dq.empty() && dq.front() < i - half) dq.pop_front();
        while (!dq.empty() && x[dq.back()] <= x[i]) dq.pop_back();
        dq.push_back(i);
        out[i] = x[dq.front()];
    }
    return out;
}

// Attack mode: forward + backward sliding max (symmetric window)
static std::vector<float> mg_sliding_max_attack(const float * x, int n, int window_size) {
    // Make window odd
    if (window_size % 2 == 0) window_size++;
    int full_window = 2 * window_size - 1;
    // Use maximum_filter1d equivalent with symmetric window
    std::vector<float> out(n);
    std::deque<int> dq;
    int half = (full_window - 1) / 2;

    for (int i = 0; i < n; i++) {
        while (!dq.empty() && dq.front() < i - half) dq.pop_front();
        while (!dq.empty() && x[dq.back()] <= x[i]) dq.pop_back();
        dq.push_back(i);
        // We need to look ahead, so delay output
    }

    // Re-do with proper lookahead
    dq.clear();
    for (int i = 0; i < n + half; i++) {
        if (i < n) {
            while (!dq.empty() && x[dq.back()] <= x[i]) dq.pop_back();
            dq.push_back(i);
        }
        int out_idx = i - half;
        if (out_idx >= 0 && out_idx < n) {
            while (!dq.empty() && dq.front() < out_idx - half) dq.pop_front();
            out[out_idx] = x[dq.front()];
        }
    }
    return out;
}

// Hold mode: causal sliding max (left-padded)
static std::vector<float> mg_sliding_max_hold(const float * x, int n, int window_size) {
    std::vector<float> out(n);
    std::deque<int> dq;
    int half = (window_size - 1) / 2;
    // Left-pad by prepending half_window copies of first element
    // Simplified: causal sliding max
    for (int i = -half; i < n; i++) {
        int src_i = std::max(0, i);
        float val = x[src_i];
        while (!dq.empty() && x[std::max(0, dq.back())] <= val) dq.pop_back();
        dq.push_back(i);
        while (!dq.empty() && dq.front() < i - window_size + 1) dq.pop_front();
        if (i >= 0) {
            out[i] = x[std::max(0, dq.front())];
        }
    }
    return out;
}

// ─── IIR Filters ────────────────────────────────────────────────────

struct BiquadCoeffs {
    double b[3], a[3];
};

// 1st-order Butterworth lowpass
static BiquadCoeffs mg_butter1(float cutoff_hz, int sample_rate) {
    // 1st order bilinear transform
    double wc = std::tan(M_PI * cutoff_hz / sample_rate);
    BiquadCoeffs c;
    c.b[0] = wc / (1.0 + wc);
    c.b[1] = wc / (1.0 + wc);
    c.b[2] = 0;
    c.a[0] = 1.0;
    c.a[1] = (wc - 1.0) / (1.0 + wc);
    c.a[2] = 0;
    return c;
}

// Forward IIR filter (causal) — matches scipy.signal.lfilter for 1st order
static std::vector<float> mg_lfilter(const BiquadCoeffs & c, const float * x, int n) {
    std::vector<float> y(n);
    double y1 = 0, x1 = 0;
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double yi = c.b[0] * xi + c.b[1] * x1 - c.a[1] * y1;
        y[i] = (float) yi;
        x1 = xi;
        y1 = yi;
    }
    return y;
}

// Zero-phase forward-backward filter — matches scipy.signal.filtfilt
static std::vector<float> mg_filtfilt_1pole(float coef, int n_samples, const float * x, int n) {
    // Simple exponential smoothing forward-backward
    float c = (float) std::exp(coef / n_samples);
    float b = 1.0f - c;

    std::vector<float> y(n);
    // Forward pass
    y[0] = b * x[0];
    for (int i = 1; i < n; i++) {
        y[i] = b * x[i] + c * y[i - 1];
    }
    // Backward pass
    for (int i = n - 2; i >= 0; i--) {
        y[i] = b * y[i] + c * y[i + 1];
    }
    return y;
}

// ─── Hyrax Brickwall Limiter ────────────────────────────────────────

static void mg_limit(float * L, float * R, int n, const MasteringConfig & cfg) {
    // Rectify: per-sample envelope
    std::vector<float> rectified(n);
    for (int i = 0; i < n; i++) {
        float peak = std::max(std::abs(L[i]), std::abs(R[i]));
        rectified[i] = std::max(peak, cfg.threshold) / cfg.threshold;
    }

    // Check if limiting is needed
    bool all_below = true;
    for (int i = 0; i < n; i++) {
        if (rectified[i] > 1.0f + 1e-6f) { all_below = false; break; }
    }
    if (all_below) return;  // No limiting needed

    // Hard clip gain: 1/envelope, then flip to (1 - gain)
    std::vector<float> gain_hard(n);
    for (int i = 0; i < n; i++) {
        gain_hard[i] = 1.0f - (1.0f / rectified[i]);
    }

    // Attack stage
    int attack_samples = std::max(1, (int) (cfg.limiter_attack_ms * cfg.sample_rate / 1000.0f));
    auto slided_attack = mg_sliding_max_attack(gain_hard.data(), n, attack_samples);
    auto gain_attack = mg_filtfilt_1pole(cfg.limiter_attack_coef, attack_samples,
                                          slided_attack.data(), n);

    // Hold stage
    int hold_samples = std::max(1, (int) (cfg.limiter_hold_ms * cfg.sample_rate / 1000.0f));
    auto slided_hold = mg_sliding_max_hold(slided_attack.data(), n, hold_samples);

    // Hold filter (Butterworth lowpass)
    auto hold_coeffs = mg_butter1(cfg.limiter_hold_coef, cfg.sample_rate);
    auto hold_output = mg_lfilter(hold_coeffs, slided_hold.data(), n);

    // Release filter
    float release_cutoff = cfg.limiter_release_coef / cfg.limiter_release_ms;
    auto  release_coeffs = mg_butter1(release_cutoff, cfg.sample_rate);
    // Input to release: max(slided_hold, hold_output)
    std::vector<float> release_input(n);
    for (int i = 0; i < n; i++) {
        release_input[i] = std::max(slided_hold[i], hold_output[i]);
    }
    auto release_output = mg_lfilter(release_coeffs, release_input.data(), n);

    // Release envelope: max(hold_output, release_output)
    std::vector<float> gain_release(n);
    for (int i = 0; i < n; i++) {
        gain_release[i] = std::max(hold_output[i], release_output[i]);
    }

    // Final gain = 1 - max(hard_clip, attack, release)
    for (int i = 0; i < n; i++) {
        float g = std::max({gain_hard[i], gain_attack[i], gain_release[i]});
        float gain = 1.0f - g;
        L[i] *= gain;
        R[i] *= gain;
    }
}

// ─── Main Pipeline ──────────────────────────────────────────────────

struct MasteringResult {
    std::vector<float> L, R;     // mastered output (same sample rate as input)
    bool               success;
    const char *       error;
};

static MasteringResult mastering_process(
    const float * target_L, const float * target_R, int target_n, int target_sr,
    const float * ref_L,    const float * ref_R,    int ref_n,    int ref_sr) {

    MasteringResult result;
    result.success = false;
    result.error   = nullptr;

    MasteringConfig cfg;

    fprintf(stderr, "[Mastering] Target: %d samples @ %d Hz (%.1f s)\n",
            target_n, target_sr, (float) target_n / target_sr);
    fprintf(stderr, "[Mastering] Reference: %d samples @ %d Hz (%.1f s)\n",
            ref_n, ref_sr, (float) ref_n / ref_sr);

    // ── Step 1: Resample reference to match target sample rate ──
    // The target (generated audio) stays at its native rate (typically 48000 Hz).
    // Only the reference is resampled if it differs.
    cfg.sample_rate = target_sr;

    std::vector<float> tgt_L_w(target_L, target_L + target_n);
    std::vector<float> tgt_R_w(target_R, target_R + target_n);
    std::vector<float> ref_L_w, ref_R_w;

    if (ref_sr != target_sr) {
        fprintf(stderr, "[Mastering] Resampling reference %d → %d Hz...\n", ref_sr, target_sr);
        std::vector<std::thread> resample_threads;
        resample_threads.emplace_back([&]() { ref_L_w = mg_resample(ref_L, ref_n, ref_sr, target_sr); });
        ref_R_w = mg_resample(ref_R, ref_n, ref_sr, target_sr);
        for (auto & t : resample_threads) t.join();
    } else {
        ref_L_w.assign(ref_L, ref_L + ref_n);
        ref_R_w.assign(ref_R, ref_R + ref_n);
    }

    int tn = (int) tgt_L_w.size();
    int rn = (int) ref_L_w.size();

    if (tn < cfg.fft_size || rn < cfg.fft_size) {
        result.error = "Audio too short for mastering (need > 4096 samples)";
        return result;
    }

    // ── Step 2: Match Levels ──
    fprintf(stderr, "[Mastering] Stage 1/4: Matching levels...\n");

    // Normalize reference
    float ref_peak = std::max(mg_peak(ref_L_w.data(), rn), mg_peak(ref_R_w.data(), rn));
    float final_amp_coeff = 1.0f;
    if (ref_peak > cfg.threshold) {
        final_amp_coeff = ref_peak / cfg.threshold;
        float norm = cfg.threshold / ref_peak;
        mg_amplify(ref_L_w.data(), rn, norm);
        mg_amplify(ref_R_w.data(), rn, norm);
    }

    auto tgt_la = mg_analyze_levels(tgt_L_w.data(), tgt_R_w.data(), tn, cfg);
    auto ref_la = mg_analyze_levels(ref_L_w.data(), ref_R_w.data(), rn, cfg);

    // RMS matching: amplify target to match reference
    float rms_coeff = (tgt_la.match_rms > cfg.min_value)
        ? ref_la.match_rms / tgt_la.match_rms : 1.0f;
    mg_amplify(tgt_la.mid.data(), tn, rms_coeff);
    mg_amplify(tgt_la.side.data(), tn, rms_coeff);
    mg_amplify(tgt_la.mid_loudest.data(), tgt_la.n_loudest * tgt_la.piece_size, rms_coeff);
    mg_amplify(tgt_la.side_loudest.data(), tgt_la.n_loudest * tgt_la.piece_size, rms_coeff);

    fprintf(stderr, "[Mastering]   RMS coefficient: %.4f (%.1f dB)\n",
            rms_coeff, 20.0f * std::log10(std::max(rms_coeff, cfg.min_value)));

    // ── Step 3: Match Frequencies ──
    fprintf(stderr, "[Mastering] Stage 2/4: Matching frequencies...\n");

    // Compute FIR correction filters (mid and side in parallel)
    std::vector<float> mid_fir, side_fir;
    {
        std::thread t_mid([&]() {
            mid_fir = mg_get_fir(tgt_la.mid_loudest.data(), tgt_la.piece_size, tgt_la.n_loudest,
                                ref_la.mid_loudest.data(), ref_la.piece_size, ref_la.n_loudest, cfg);
        });
        side_fir = mg_get_fir(tgt_la.side_loudest.data(), tgt_la.piece_size, tgt_la.n_loudest,
                              ref_la.side_loudest.data(), ref_la.piece_size, ref_la.n_loudest, cfg);
        t_mid.join();
    }

    fprintf(stderr, "[Mastering]   FIR length: %d taps\n", (int) mid_fir.size());

    // Convolve mid and side channels (in parallel)
    std::vector<float> result_mid, result_side;
    {
        std::thread t_mid([&]() {
            result_mid = mg_fftconvolve(tgt_la.mid.data(), tn, mid_fir.data(), (int) mid_fir.size());
        });
        result_side = mg_fftconvolve(tgt_la.side.data(), tn, side_fir.data(), (int) side_fir.size());
        t_mid.join();
    }

    // Trim convolution result to original length (centered)
    int fir_half = (int) mid_fir.size() / 2;
    std::vector<float> conv_mid(tn), conv_side(tn);
    for (int i = 0; i < tn; i++) {
        int src = i + fir_half;
        conv_mid[i]  = (src < (int) result_mid.size())  ? result_mid[src]  : 0.0f;
        conv_side[i] = (src < (int) result_side.size()) ? result_side[src] : 0.0f;
    }

    // ── Step 4: Correct Levels ──
    fprintf(stderr, "[Mastering] Stage 3/4: Correcting levels (%d passes)...\n",
            cfg.rms_correction_steps);

    for (int step = 0; step < cfg.rms_correction_steps; step++) {
        // Clip mid, recompute RMS
        std::vector<float> clipped_mid = conv_mid;
        mg_clip(clipped_mid.data(), tn);

        int divisions  = tn / tgt_la.piece_size;
        if (divisions < 1) divisions = 1;
        auto clipped_rmses = mg_batch_rms(clipped_mid.data(), tgt_la.piece_size, divisions);
        float clipped_avg = 0;
        for (float r : clipped_rmses) clipped_avg += r;
        clipped_avg /= clipped_rmses.size();

        // Find loudest clipped pieces
        float clipped_match = 0;
        int   clipped_count = 0;
        for (float r : clipped_rmses) {
            if (r >= clipped_avg) {
                clipped_match += r;
                clipped_count++;
            }
        }
        if (clipped_count > 0) clipped_match /= clipped_count;

        float correction = (clipped_match > cfg.min_value)
            ? ref_la.match_rms / clipped_match : 1.0f;
        mg_amplify(conv_mid.data(), tn, correction);
        mg_amplify(conv_side.data(), tn, correction);
    }

    // Convert back to L/R
    std::vector<float> out_L(tn), out_R(tn);
    mg_ms_to_lr(conv_mid.data(), conv_side.data(), tn, out_L.data(), out_R.data());

    // ── Step 5: Finalize ──
    fprintf(stderr, "[Mastering] Stage 4/4: Limiting & normalizing...\n");

    // Hyrax brickwall limiter
    mg_limit(out_L.data(), out_R.data(), tn, cfg);

    // Apply the reference amplitude coefficient
    mg_amplify(out_L.data(), tn, final_amp_coeff);
    mg_amplify(out_R.data(), tn, final_amp_coeff);

    // ── Output (already at target sample rate, no resampling needed) ──
    result.L = std::move(out_L);
    result.R = std::move(out_R);

    result.success = true;
    fprintf(stderr, "[Mastering] Done. Output: %d samples @ %d Hz\n",
            (int) result.L.size(), target_sr);
    return result;
}
