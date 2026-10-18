/*
Minimal GPU-only benchmark for the Apple Metal on-the-fly BEM operators.

Unlike metal_onthefly_test this program does not compute a CPU reference; it
only builds the four on-the-fly matrices on the GPU and times a number of
apply() calls, reporting the minimum and median wall-clock time. This keeps the
GPU busy back-to-back and avoids the (much slower) CPU reference dominating the
runtime, which makes it suitable for comparing kernel versions/tile sizes.

Usage:
  metal_onthefly_bench <mesh_file> <n_timesteps> <gpu_kernel_version> <reps>
*/

#include "besthea/besthea.h"
#include "besthea/besthea_metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::bem::onthefly;

template<class kernel_type, class test_space_type, class trial_space_type>
void bench_operator(const std::string & label, kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  const uniform_spacetime_tensor_mesh_gpu & st_mesh_gpu, lo n_timesteps,
  int gpu_kernel_version, int reps) {
  uniform_spacetime_be_matrix_onthefly_gpu<kernel_type, test_space_type, trial_space_type>
    mat(kernel, test_space, trial_space, st_mesh_gpu, 4, 4, gpu_kernel_version);

  lo dim_dom = mat.get_dim_domain();
  lo dim_rng = mat.get_dim_range();

  block_vector x(n_timesteps, dim_dom);
  for (lo b = 0; b < n_timesteps; ++b) {
    for (lo i = 0; i < dim_dom; ++i) {
      x.get_block(b).set(i, std::sin((b + 1) * 0.7 + i * 0.3));
    }
  }
  block_vector y(n_timesteps, dim_rng);

  // Warm-up (pipeline creation, first allocation, ...).
  mat.apply(x, y);

  std::vector<double> times;
  times.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    mat.apply(x, y);
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(times.begin(), times.end());
  double mn = times.front();
  double md = times[times.size() / 2];

  if (std::getenv("BESTHEA_BENCH_VERBOSE") != nullptr) {
    std::cout << "  " << label << " all:";
    for (double t : times) std::cout << " " << std::fixed << std::setprecision(5) << t;
    std::cout << "\n";
  }

  std::cout << "BENCH " << label << " version=" << gpu_kernel_version
            << " min=" << std::fixed << std::setprecision(5) << mn
            << " median=" << md << " n=" << reps << "\n";
}

int main(int argc, char ** argv) {
  std::string mesh_file = "examples/mesh_files/cube_192.txt";
  if (argc > 1) mesh_file = argv[1];
  lo n_timesteps = 32;
  if (argc > 2) n_timesteps = std::atoi(argv[2]);
  int gpu_kernel_version = 3;
  if (argc > 3) gpu_kernel_version = std::atoi(argv[3]);
  int reps = 10;
  if (argc > 4) reps = std::atoi(argv[4]);

  sc end_time = 1.0;
  sc alpha = 0.5;

  triangular_surface_mesh surface_mesh(mesh_file);
  uniform_spacetime_tensor_mesh st_mesh(surface_mesh, end_time, n_timesteps);
  uniform_spacetime_tensor_mesh_gpu st_mesh_gpu(st_mesh);

  uniform_spacetime_be_space<basis_tri_p0> space_p0(st_mesh);
  uniform_spacetime_be_space<basis_tri_p1> space_p1(st_mesh);

  spacetime_heat_sl_kernel_antiderivative kernel_sl(alpha);
  spacetime_heat_dl_kernel_antiderivative kernel_dl(alpha);
  spacetime_heat_adl_kernel_antiderivative kernel_adl(alpha);
  spacetime_heat_hs_kernel_antiderivative kernel_hs(alpha);

  std::cout << "mesh=" << mesh_file << " elements=" << surface_mesh.get_n_elements()
            << " timesteps=" << n_timesteps
            << " version=" << gpu_kernel_version << " reps=" << reps << "\n";

  bench_operator("V", kernel_sl, space_p0, space_p0, st_mesh_gpu, n_timesteps, gpu_kernel_version, reps);
  bench_operator("K", kernel_dl, space_p0, space_p1, st_mesh_gpu, n_timesteps, gpu_kernel_version, reps);
  bench_operator("K'", kernel_adl, space_p1, space_p0, st_mesh_gpu, n_timesteps, gpu_kernel_version, reps);
  bench_operator("D", kernel_hs, space_p1, space_p1, st_mesh_gpu, n_timesteps, gpu_kernel_version, reps);

  return 0;
}
