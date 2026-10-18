# BESTHEA — Apple Metal performance work: log and plan

Author: performance session notes
Date: 2026-10-18
Branch: `master`
Scope: the Apple Metal port of the on-the-fly space–time BEM matrix kernels
(`src/uniform_spacetime_be_matrix_onthefly_gpu.mm` +
`src/uniform_spacetime_be_matrix_onthefly_metal.metal`).

This document records **what has been done** (with commits, measurements and
methodology so the numbers can be re-derived) and **what still needs to be
done**, in priority order.

---

## 1. Context

The project is BESTHEA, a space–time boundary element solver for the heat
equation, originally CUDA, ported to Apple Metal. On GPUs the dominant cost is
the assembly of the on-the-fly BEM matrix operators (they are never stored; the
matrix blocks are applied on the fly). The four operators are:

| op | name | spaces |
|----|------|--------|
| V  | single layer potential (sl)                | p0 → p0 |
| K  | double layer potential (dl)                | p0 → p1 |
| K′ | adjoint double layer potential (adl)       | p1 → p0 |
| D  | hypersingular potential (hs)               | p1 → p1 |

All GPU arithmetic is `float` (Apple GPUs have no FP64); `sc` is `double` on
the host only. The Metal backend hardcodes `n_gpus = 1`.

Measured efficiency is ~0.5–1 TFLOPS against an M4-class peak of ~4–5 TFLOPS,
i.e. roughly 15–20 % of peak.

---

## 2. Environment / ground rules

- Platform: Apple Silicon (M4-class), macOS, `darwin`.
- MSL: `-std=metal3.0`, compiled with `-O3 -ffast-math -ffp-contract=fast`.
- `maxThreadgroupMemoryLength = 32768` bytes on the target GPU.
- Build dir: `build/`. Shaders build to `build/besthea_shaders.metallib`
  (target `besthea_shaders`); it is reloaded by the host at process start.
- Binaries used below:
  - `build/examples/metal_onthefly_test/metal_onthefly_test <mesh> <n_timesteps> <gpu_kernel_version>`
  - `build/examples/metal_onthefly_bench/metal_onthefly_bench <mesh> <n_timesteps> <gpu_kernel_version> <reps>`
- **Benchmark methodology notes** (important, learned the hard way):
  - Use **min-of-N**, not median. Thermal ramp inflates medians and late reps.
  - The **first run after idle is a cold outlier** (fastest); do not report a
    single run. Alternate configurations and take the min over all reps/rounds.
  - Comparisons must use the **same timestep count** and the same binary.
  - The host `.mm` is unchanged by shader-only work, so a **shader A/B can be
    done by swapping `build/besthea_shaders.metallib`** (see §6).

---

## 3. What has been done

### 3.1 Round 1 — `bdf06cb` "perf(metal): 32-bit indices, vectorized conversions, shader -O3/-ffast-math"

- **32-bit indices in the shader.** All 175 `long` tokens in the `.metal`
  kernel were replaced with `int` (the host still keeps `lo * d_element_nodes`
  for CUDA compatibility and adds a separate `int * d_element_nodes_int` for the
  Metal path). Accepted constraint: `dim × n_blocks < 2³¹`.
- **Vectorized host↔device conversion** using OpenMP
  (`#pragma omp parallel for`) to avoid linking Accelerate/vDSP.
- **Shader build flags**: `metal -std=metal3.0 -O3 -ffast-math -ffp-contract=fast`
  (`CMakeLists.txt`).

Files: `CMakeLists.txt`, `include/besthea/uniform_spacetime_tensor_mesh_gpu.h`,
`src/uniform_spacetime_be_matrix_onthefly_gpu.mm`,
`src/uniform_spacetime_be_matrix_onthefly_metal.metal`,
`src/uniform_spacetime_tensor_mesh_gpu.mm`.

### 3.2 Round 2 — `2e48f72` "perf(metal): add tiled version 3 on-the-fly kernels"

Ported the CUDA "version 3" tiled, on-the-fly temporal-block kernels:

- **2-D tile** `tpbx` (test) × `tpby` (trial) per threadgroup, with **both**
  tiles' quadrature nodes staged in threadgroup memory (shared across the tile).
- Geometry evaluation is now amortized: ~`n_elems²·(1/tpbx + 1/tpby)` geometry
  evals instead of `n_elems²`.
- Block rows are distributed across the `y` dimension with stride `tpby`; p0
  test spaces need no atomics, p1 test spaces (adl, hs) use them.
- Tile shape is a **template parameter encoded in the kernel name suffix**
  `_t<TX>x<TY>`, so a stale host/shader mismatch is impossible.
  Tiles instantiated: `4x32`, `8x32`, `2x32`, `8x16`. Default **4x32**.
  Runtime override via env `BESTHEA_VER3_TILE` (e.g. `8x32`).
- Version 3 became the default (`gpu_kernel_version_ = 3`); version 2 retained.
- Added a GPU-only benchmark example `examples/metal_onthefly_bench`.
- Host: version selector, 2-D dispatch `(tpbx, tpby, 1)`, grid `ceil(count/tpbx)`,
  new buffer 13 = `i_tst_count` used for masking.

**Results** (min-of-N):
- `cube_192` nt=128, v3 vs v2: **V −44 %, K −25 %, K′ −32 %, D −21 %**.
- `scaled_crankshaft_11k` nt=8, v3 vs v2: V +2 %, K −9 %, K′ −5 %, D −2 %.

**Tile-shape findings:** 8×8 and 16×8 are much worse; 4×32 and 8×32 best on
`cube_192`; 2×32 best on tiny meshes (`cube_24`). Default 4×32. Threadgroup
usage for hs at 4×32/qo5 ≈ 24 KB (the board limit is 32 KB).

### 3.3 P0.4 — evaluated and **dismissed**

"Precompute per-element quadrature nodes once on device" was evaluated and
dismissed as subsumed: v3 already shares trial nodes across each `tpbx` tile,
reducing geometry evals to `n_elems²·(1/tpbx + 1/tpby)`, and geometry is a small
fraction of total arithmetic.

### 3.4 Metal 4 migration — declined

The project uses MSL 3.0 and the classic `MTLComputeCommandEncoder`, not
Metal 4. Migrating would not improve throughput (the workload is not bound by
the command-encoder API) and would raise the deployment target. Offered but not
done: bump to `-std=metal3.2` / `MTLLanguageVersion3_2`.

### 3.5 Round 3 — `f525563` "perf(metal): fuse erf/exp and hoist per-block heat-kernel constants"

Motivation (found by reading the kernel bodies):

- `metal_erf(x)` (Abramowitz & Stegun 7.1.26) internally computes
  `exp(-x²)`, **and** every caller separately computed the *same* Gaussian
  `exp(-norm²/(4τ·α))`. So the expensive exponential was evaluated **twice per
  quadrature point**, together with a duplicate `norm²`.
- The `τ`-only factors (`1/(2√τ√α)`, `1/(4τα)`, `1/(4πα)`, `1/(8πα²)`, …) were
  recomputed inside the per-point quadrature loop.

Changes in `src/uniform_spacetime_be_matrix_onthefly_metal.metal`:

1. Replaced `metal_erf` with `metal_erf_nonneg(u, e)` — same A&S polynomial, but
   takes the already-computed `e = exp(-u²)` and drops the sign/abs/`u>5` branch
   (`u ≥ 0` always).
2. Added `d_heat_kernel_eval_core()`, which computes `norm`, `1/norm`, `u`,
   a **single** `exp(-u²)` and `erf(u)` once per quadrature point and shares them
   with all four operator kernels.
3. Added `heat_kernel_delta_constants_float` + `d_heat_kernel_delta_constants()`,
   computed **once per (element pair, temporal block)** rather than per point.
4. Replaced divisions with a single reciprocal (`1/norm`) and multiplications
   (`τ/norm² → τ·inv_norm²`, etc.).

Accuracy is unchanged (still A&S 7.1.26, |err| ≤ 1.5e-7):

| case | V | K | K′ | D |
|------|---|---|----|----|
| cube_12 nt=4         | 7.3e-6 | 1.8e-6 | 1.4e-6 | 1.3e-6 |
| cube_24 nt=8         | 1.5e-5 | 3.3e-6 | 3.1e-6 | 4.5e-6 |
| cube_192 nt=12       | 1.4e-5 | 6.2e-6 | 1.6e-6 | 5.6e-6 |
| crankshaft_11k nt=4  | 2.7e-7 | 1.9e-7 | 1.6e-7 | 2.3e-7 |

v2 also still passes (it shares the same kernel evaluator).

**Results** (min-of-N):

`cube_192` nt=128, v3, before (`2e48f72`) → after (`f525563`):

| op | before | after | Δ |
|----|--------|-------|---|
| V  | 7.67 ms | 7.17 ms | −6 % |
| K  | 14.46 ms | 13.27 ms | −8 % |
| K′ | 10.60 ms | 10.11 ms | −5 % |
| D  | 20.06 ms | 17.56 ms | −13 % |

v2 also improves (shared evaluator): V −2 %, K −2 %, K′ −2 %, D −11 %.

`scaled_crankshaft_11k` nt=8: **neutral within noise** (the one apparent "−44 %"
before-run was a cold-start artifact — see the methodology note).

---

## 4. Profiling the remaining cost (round 3, step 0)

Goal: find the true bottleneck before investing further in the kernel evaluator.

Method: build **throwaway metallibs** in which one piece of the kernel is
stubbed out, keep everything else identical, and A/B the V operator on
`cube_192` nt=128 (min-of-N). Stubs used:

```metal
// m_nobody : remove the whole per-point kernel body (deltas + math), keep loop/geometry
value += w[i_quad];                              // instead of kernel(...) * w[i_quad]

// m_sqrt   : remove sqrt + the 1/n division
const float n     = norm2;                       // instead of metal::sqrt(norm2)
const float inv_n = norm2;                       // instead of 1.0f / n

// mock (m_notrans) : remove erf + exp from the shared core
const float e    = -norm2 * dc.inv_4_tau_alpha;  // instead of metal::exp(-norm2 * dc.inv_4_tau_alpha)
*erf_value       = n * dc.inv_2_sqrt_tau_sqrt_alpha;  // instead of metal_erf_nonneg(...)

// m_nogather : remove the x_perm gather in the contraction
const float x_val = 1.0f;                        // instead of x_perm[block_col + ld_x_perm * i_trl_j]
```

**Attribution (V, cube_192 nt=128, v3, min-of-N):**

| variant | V time | Δ vs committed |
|---------|--------|----------------|
| committed (`f525563`)            | 7.20 ms | — |
| remove `erf` + `exp`             | —       | **−14 %** |
| remove `sqrt` + `1/x`            | —       | **−8 %** |
| remove whole per-point body      | 4.77 ms | **−33 %** |
| remove `x_perm` gather           | 6.36 ms | **−12 %** |

Interpretation:

- **Per-point heat-kernel math ≈ 33 %** of V (erf+exp 14 %, sqrt+div 8 %, the
  remaining FMAs/delta loads ≈ 11 %).
- The contraction `x_perm` gather alone ≈ 12 %.
- The remaining **≈ 55 %** is structural: geometry taping, tile/block loop
  control, `threadgroup_barrier`s, the contraction's own FMAs + threadgroup
  reads, and the scattered `y` accumulate.

The dominant structural piece is the **temporal-convolution contraction**: per
thread it is ~`n_blocks²/2` FMAs plus a global gather and a scattered
accumulate, quadratic in `n_blocks` — the same order as the kernel evaluation
itself.

### Conclusion of the profiling

- The transcendentals are **not** the bottleneck. erf+exp is at most ~14 % of V,
  and the fused version already recovered about half of it. Cheaper `erf` and
  vectorising the kernel eval have a ceiling of a few percent. **Do not pursue
  them further.**
- The real levers are structural (contraction, geometry, memory, occupancy) or
  algorithmic (fewer pairs).

---

## 5. What still needs to be done (prioritised)

### P0 — cheap, do first

1. **Fresh tile / threadgroup sweep.** The tile sweep predates both the fused
   evaluator and the int32 change. Re-run it (env `BESTHEA_VER3_TILE`, and the
   threadgroup sizes) and re-pick the default. Cheap and may interact with the
   evaluator change. Also check `MTLComputePipelineState` limits
   (`maxTotalThreadsPerThreadgroup`) and register pressure after the change.
2. **Check occupancy / register pressure**, especially for the hs kernel, which
   holds many arrays (`test_curls[9]`, `trial_curls[9]`, `curl_dot[9]`,
   `values_out[9]`). Lower registers / smaller threadgroups may raise occupancy.

### P1 — the real structural levers

3. **Contraction cost.** The `x_perm` gather alone is 12 %, plus its FMAs.
   Candidate approaches:
   - stage the `x_perm` tile in **threadgroup memory** so the per-delta gather
     hits shared memory instead of global;
   - process the contraction with **`float4`** ops;
   - consider **`simdgroup_matrix`** for the temporal Toeplitz-like contraction;
   - reduce `shmem_matrix_vals` traffic between the evaluation and contraction
     phases.
4. **Geometry amortisation.** Geometry + loop + barrier overhead is part of the
   ~55 %. Larger tiles regressed in the last sweep, but that was pre-change; see
   P0.1.
5. **Memory layout of `y`.** The accumulate `y[block_row * ld_y + row]` is a
   scattered RMW; for p1 test spaces it also uses atomics. Investigate
   conflict-free layouts / per-threadgroup staging + one write.

### P2 — algorithmic (largest potential, changes the method)

6. **Evaluate fewer pairs.** The cost is `O(n_elems² · n_blocks)`. The heat
   kernel decays in τ, and spatially distant clusters contribute negligibly, so
   **temporal truncation** and **spatial clustering / FMM-like** schemes can cut
   the count rather than the constant. The repo already has FMM experiments to
   build on. This is the only lever that changes the exponent/leading constant.
   It changes results, so it needs an error budget and a tolerance story.

### P3 — optional / low value

7. **MSL 3.2** (`-std=metal3.2`, `MTLLanguageVersion3_2`): offered, not taken.
   Low expected value.
8. **Metal 4 migration**: declined (no throughput benefit; raises deployment
   target).
9. **Further micro-optimisation of the per-eval math** (cheaper erf, rsqrt,
   float4 in the evaluator): capped at a few percent by §4 — skip.

### Housekeeping

10. `temp_geometry_files/` is untracked and predates this work; decide whether to
    keep, ignore, or remove it.
11. Consider committing this document (currently created, not committed).

---

## 6. How to reproduce

### Build

```sh
cmake --build build --target besthea_shaders metal_onthefly_test metal_onthefly_bench
```

### Correctness

```sh
./build/examples/metal_onthefly_test/metal_onthefly_test examples/mesh_files/cube_12.txt   4 3
./build/examples/metal_onthefly_test/metal_onthefly_test examples/mesh_files/cube_24.txt   8 3
./build/examples/metal_onthefly_test/metal_onthefly_test examples/mesh_files/cube_192.txt 12 3
./build/examples/metal_onthefly_test/metal_onthefly_test examples/mesh_files/scaled_crankshaft_11k.txt 4 3
# v2 sanity:
./build/examples/metal_onthefly_test/metal_onthefly_test examples/mesh_files/cube_192.txt 8 2
```

Expect all four operators `PASS` with relative errors in the 1e-7 … 1e-5 range.

### Benchmark

```sh
# min-of-N; run several times and take the min
./build/examples/metal_onthefly_bench/metal_onthefly_bench examples/mesh_files/cube_192.txt 128 3 10
./build/examples/metal_onthefly_bench/metal_onthefly_bench examples/mesh_files/scaled_crankshaft_11k.txt 8 3 6
# BESTHEA_BENCH_VERBOSE=1 dumps every rep
```

### Shader A/B without rebuilding the host

```sh
cp build/besthea_shaders.metallib /tmp/after.metallib
# ... build the other variant ...
cp /tmp/other.metallib build/besthea_shaders.metallib   # swap before running
```

Note: CMake's custom command for the shader can be skipped on a same-second
timestamp race; use `rm -f build/besthea_shaders.metallib` before rebuilding to
force it.

### Tile override

```sh
BESTHEA_VER3_TILE=8x32 ./build/examples/metal_onthefly_bench/metal_onthefly_bench examples/mesh_files/cube_192.txt 128 3 10
```

---

## 7. Commits and files touched

| commit | message |
|--------|---------|
| `bdf06cb` | perf(metal): 32-bit indices, vectorized conversions, shader -O3/-ffast-math |
| `2e48f72` | perf(metal): add tiled "version 3" on-the-fly kernels |
| `f525563` | perf(metal): fuse erf/exp and hoist per-block heat-kernel constants |

Key files:

- `src/uniform_spacetime_be_matrix_onthefly_metal.metal` — v2 + v3 tiled kernels,
  threadgroup geometry/contribution helpers, fused heat-kernel evaluator
  (`metal_erf_nonneg`, `heat_kernel_delta_constants_float`,
  `d_heat_kernel_delta_constants`, `d_heat_kernel_eval_core`), hs curl hoisting,
  specialisations.
- `src/uniform_spacetime_be_matrix_onthefly_gpu.mm` — version selectors,
  `resolve_ver3_tile` / `ver3_kernel_name`, dispatch, `MTLLanguageVersion3_0`.
- `include/besthea/uniform_spacetime_be_matrix_onthefly_gpu.h` — default
  `gpu_kernel_version_ = 3`.
- `src/uniform_spacetime_be_matrix_onthefly_gpu.cu` — CUDA reference (versions
  1–4) used for the port.
- `examples/metal_onthefly_test/` — correctness harness (3rd arg = version,
  default 3).
- `examples/metal_onthefly_bench/` — GPU-only benchmark (args: mesh, timesteps,
  version, reps).
- `CMakeLists.txt` / `examples/CMakeLists.txt` — shader flags and example targets.
- `examples/mesh_files/scaled_crankshaft_11k.txt` — real large mesh (11k elems,
  5363 nodes) used for validation and benchmarking.

---

## 8. TL;DR

- Three perf commits landed: int32 + vectorised conversion + `-O3/-ffast-math`
  (`bdf06cb`), tiled v3 kernels (`2e48f72`, v3 vs v2 up to −44 %), and the fused
  heat-kernel evaluator (`f525563`, cube_192 v3 up to −13 %, neutral on the big
  mesh).
- Profiling shows the kernel is **not** transcendental-bound: erf+exp ≤ 14 % of
  V, sqrt+div ≈ 8 %, the whole per-point body ≈ 33 %. The remaining ≈ 67 % is
  the **temporal-convolution contraction**, geometry, loop and memory overhead.
- Next: (P0) re-sweep tiles/occupancy; (P1) attack the contraction and memory
  layout; (P2) the only big lever left is algorithmic — evaluate fewer pairs.
