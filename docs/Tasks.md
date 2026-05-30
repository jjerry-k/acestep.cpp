# Tasks — HOT-Step feature integration

Porting selected HOT-Step-CPP engine features into this engine (`acestep.cpp`).
All work is on branch **`feature/hot-step-integration`** (off `fdee882`).

> HOT-Step's engine is a fork synced to acestep.cpp@`31cc9ea` (Apr 2026); this
> tree is ~49 commits ahead, so HOT files rarely drop in cleanly.
> Workspace paths below (`results/…`) are relative to the parent research dir.

## Guiding principles

- **Never edit `src/dit.h` or `src/dit-graph.h`.** They are the verified
  generation path behind `dit_ggml_generate`. `dit.h` is ~77% diverged from HOT.
  Every port is **additive** (new files / new consumers).
- **Default OFF, opt-in.** Each feature is gated by a request field or env toggle;
  with it off the C++ generation path stays **byte-identical**.
- **Regression gate:** regenerate `results/baseline.json` on GPU 6 and compare the
  WAV md5 against `results/_gencheck/baseline00.wav` after any change that could
  touch the generation path.
- **License:** acestep.cpp is MIT — GPL `md_*` plugins are excluded.

## Completed (committed)

| # | Commit | Task | Verification |
|---|--------|------|--------------|
| 1 | `f1b050d` | Lua plugin runtime (solvers/schedulers/guidance) + dr_flac FLAC input | baseline byte-identical; apg.lua byte-identical; dpm3m.lua valid audio |
| 2 | `d64d4f8` | Multi-eval Lua solvers via `model_fn` (heun/rk4/dopri5/…) | heun (2 NFE) valid audio; single-eval paths byte-identical |
| 3 | `751e593` | Request-driven plugin selection (`lua_plugins`/`scheduler`/`guidance_mode`) | request {cosine+heun+cfg_pp} dispatched via Lua; default byte-identical |
| 4 | `3a3c416` | [A] CPU audio post-proc: `ace-master` tool + opt-in denoiser + spectral lifter | `ace-master` masters a WAV; denoise/lift run; off → byte-identical |
| 5 | `e8a58c8` | Repaint reinject mode (per-step inject + crossfade blend) | reinject+blend runs, hard splice skipped, valid audio; off → byte-identical |
| 6 | `6fdcc77` | [C] LRC alignment **code only** (compile-guard; not wired) | `test-alignment` compiles/links; gen path unchanged |

### Usage of the new features
- **Lua plugins (per request):** `"lua_plugins": true` + optional `"solver"`,
  `"scheduler"`, `"guidance_mode"`. Dev override: `ACE_LUA_SOLVER`/`SCHEDULER`/`GUIDANCE=1`.
  Plugins dir auto-resolved from the executable (`<exe>/../plugins`) or `ACE_PLUGINS_DIR`.
- **dr_flac:** `ace-synth --src-audio foo.flac` (FLAC input decode).
- **Mastering:** `ace-master --target in.wav --reference ref.wav --output out.wav [--pcm24|--pcm32f]`.
- **Denoise / spectral lift (per request, post-VAE):** `"denoise_strength": 0.5`
  (+`denoise_smoothing`/`denoise_mix`) and/or `"spectral_lift": true`.
- **Repaint reinject (per request, repaint task):** `"repaint_injection_ratio": 0.5`,
  `"repaint_crossfade_frames": 8` + `ace-synth --src-audio …` (0 = default splice).

## Decisions / not done

- **[D] runtime-LoRA — skipped.** HOT applies the LoRA delta *inside the forward*
  (`dit_ggml_linear_lora` per projection in `dit-graph.h`, un-fused QKV + `DiTLoRA`
  in `dit.h`, sideband-gated). A faithful port would edit the verified path. ACE
  already has working **static** adapter merge (`adapter`/`adapter_scale`/`--adapters`).
- **[B] SuperSep (stem separation) — not requested.** Heavy ONNX-Runtime dependency,
  a different task from generation; would be a standalone tool if ever wanted.
- **[E] VST3 host / custom server — not requested.** DAW-plugin niche; `ace-server`
  already exists.

## Known limitations / issues

- **Multi-eval solver + APG momentum:** multi-eval solvers call `model_fn` several
  times per step; each call updates the APG momentum buffer, so smoothing becomes
  per-evaluation instead of per-step (matches HOT, theoretically imperfect). The
  default `dpm3m` (1-NFE) is unaffected.
- **Lua non-determinism:** Lua computes in double, C++ in float → Lua solvers/
  schedulers are not byte-identical to the C++ path (equivalent quality). Use the
  C++ default for exact reproducibility.
- **Unwired modes:** `owns_loop` solvers (storm) and `post_step` guidance (cfg_mp)
  auto-fall-back to native (not broken). 0 postprocess plugins (GPL excluded).
- **Denoiser normalization order:** ACE denoises then `audio_write` normalizes;
  HOT normalizes first → results valid but can differ slightly.
- **Untested:** full `ace-lm → ace-synth` round-trip of the new request fields
  (verified via direct synth injection); batch N>1 + new features; boundary values.
- **Plugin path** resolution uses `/proc/self/exe` (Linux); other platforms need
  `ACE_PLUGINS_DIR`.

## Next steps (deferred)

- **Wire LRC alignment as opt-in post-generation/cover output.** The code compiles
  (commit `6fdcc77`). To wire: run a separate alignment forward (`dit_alignment_extract`),
  populate `AlignmentConfig` (layer/head targets), feed scores to `lrc-alignment.h`,
  emit an opt-in `.lrc` (e.g. request `"align_lyrics": true`), and spot-check the
  timestamps against the actual vocal positions. Keep it off the generation path.
- Optional hardening of the known limitations above (per-step APG for multi-eval;
  lm→synth round-trip test; batch/boundary coverage).
