#include "besthea/besthea.h"
#include "besthea/besthea_metal.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace besthea::mesh;
using namespace besthea::linear_algebra;
using namespace besthea::bem;
using namespace besthea::bem::onthefly;

int main(int argc, char ** argv) {
  std::cout << "=========================================================\n";
  std::cout << "  BESTHEA Apple Silicon Metal On-the-Fly BEM Test\n";
  std::cout << "=========================================================\n\n";

  std::string mesh_file = "examples/mesh_files/cube_24.txt";
  if (argc > 1) {
    mesh_file = argv[1];
  }

  std::cout << "Loading spatial surface mesh from: " << mesh_file << "\n";
  triangular_surface_mesh surface_mesh(mesh_file);
  std::cout << "  Surface elements: " << surface_mesh.get_n_elements()
            << ", nodes: " << surface_mesh.get_n_nodes() << "\n";

  lo n_timesteps = 4;
  if (argc > 2) {
    n_timesteps = std::atoi(argv[2]);
  }
  sc end_time = 1.0;
  sc alpha = 0.5;

  std::cout << "Creating spacetime tensor mesh (timesteps: " << n_timesteps
            << ", end_time: " << end_time << ")...\n";
  uniform_spacetime_tensor_mesh st_mesh(surface_mesh, end_time, n_timesteps);

  std::cout << "Initializing Metal GPU spacetime mesh...\n";
  uniform_spacetime_tensor_mesh_gpu st_mesh_gpu(st_mesh);
  std::cout << "  Metal GPUs detected: " << st_mesh_gpu.get_n_gpus() << "\n\n";

  // Spaces
  uniform_spacetime_be_space<basis_tri_p0> space_p0(st_mesh);
  uniform_spacetime_be_space<basis_tri_p1> space_p1(st_mesh);

  // Kernel antiderivatives
  spacetime_heat_sl_kernel_antiderivative kernel_sl(alpha);
  spacetime_heat_dl_kernel_antiderivative kernel_dl(alpha);
  spacetime_heat_adl_kernel_antiderivative kernel_adl(alpha);
  spacetime_heat_hs_kernel_antiderivative kernel_hs(alpha);

  bool all_passed = true;

  // -------------------------------------------------------------
  // Test 1: Single Layer Potential (V: p0 -> p0)
  // -------------------------------------------------------------
  {
    std::cout << "--- Testing Operator 1: Single Layer Potential (V, p0 -> p0) ---\n";
    uniform_spacetime_be_matrix_onthefly_cpu<
      spacetime_heat_sl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p0>,
      uniform_spacetime_be_space<basis_tri_p0>> V_cpu(kernel_sl, space_p0, space_p0, 4, 4);

    uniform_spacetime_be_matrix_onthefly_gpu<
      spacetime_heat_sl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p0>,
      uniform_spacetime_be_space<basis_tri_p0>> V_gpu(kernel_sl, space_p0, space_p0, st_mesh_gpu, 4, 4, 2);

    lo dim_dom = V_cpu.get_dim_domain();
    lo dim_rng = V_cpu.get_dim_range();
    block_vector x(n_timesteps, dim_dom);
    for (lo b = 0; b < n_timesteps; ++b) {
      for (lo i = 0; i < dim_dom; ++i) {
        x.get_block(b).set(i, std::sin((b + 1) * 0.7 + i * 0.3));
      }
    }

    block_vector y_cpu(n_timesteps, dim_rng);
    block_vector y_gpu(n_timesteps, dim_rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    V_cpu.apply(x, y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    V_gpu.apply(x, y_gpu);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t1 - t0).count();

    block_vector diff(y_gpu);
    diff.add(y_cpu, -1.0);
    sc norm_cpu = y_cpu.norm();
    sc rel_err = diff.norm() / (norm_cpu > 1e-12 ? norm_cpu : 1.0);

    std::cout << "  CPU time:   " << std::fixed << std::setprecision(5) << cpu_time << " s\n";
    std::cout << "  Metal time: " << std::fixed << std::setprecision(5) << gpu_time << " s\n";
    std::cout << "  Speedup:    " << (cpu_time / (gpu_time > 0 ? gpu_time : 1e-6)) << "x\n";
    std::cout << "  ||y_metal - y_cpu|| / ||y_cpu||: " << std::scientific << rel_err << "\n";

    if (rel_err < 1e-3) {
      std::cout << "  Result: PASS [V]\n\n";
    } else {
      std::cout << "  Result: FAIL [V]\n\n";
      all_passed = false;
    }
  }

  // -------------------------------------------------------------
  // Test 2: Double Layer Potential (K: p1 -> p0)
  // -------------------------------------------------------------
  {
    std::cout << "--- Testing Operator 2: Double Layer Potential (K, p1 -> p0) ---\n";
    uniform_spacetime_be_matrix_onthefly_cpu<
      spacetime_heat_dl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p0>,
      uniform_spacetime_be_space<basis_tri_p1>> K_cpu(kernel_dl, space_p0, space_p1, 4, 4);

    uniform_spacetime_be_matrix_onthefly_gpu<
      spacetime_heat_dl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p0>,
      uniform_spacetime_be_space<basis_tri_p1>> K_gpu(kernel_dl, space_p0, space_p1, st_mesh_gpu, 4, 4, 2);

    lo dim_dom = K_cpu.get_dim_domain();
    lo dim_rng = K_cpu.get_dim_range();
    block_vector x(n_timesteps, dim_dom);
    for (lo b = 0; b < n_timesteps; ++b) {
      for (lo i = 0; i < dim_dom; ++i) {
        x.get_block(b).set(i, std::cos((b + 1) * 0.5 + i * 0.4));
      }
    }

    block_vector y_cpu(n_timesteps, dim_rng);
    block_vector y_gpu(n_timesteps, dim_rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    K_cpu.apply(x, y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    K_gpu.apply(x, y_gpu);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t1 - t0).count();

    block_vector diff(y_gpu);
    diff.add(y_cpu, -1.0);
    sc norm_cpu = y_cpu.norm();
    sc rel_err = diff.norm() / (norm_cpu > 1e-12 ? norm_cpu : 1.0);

    std::cout << "  CPU time:   " << std::fixed << std::setprecision(5) << cpu_time << " s\n";
    std::cout << "  Metal time: " << std::fixed << std::setprecision(5) << gpu_time << " s\n";
    std::cout << "  Speedup:    " << (cpu_time / (gpu_time > 0 ? gpu_time : 1e-6)) << "x\n";
    std::cout << "  ||y_metal - y_cpu|| / ||y_cpu||: " << std::scientific << rel_err << "\n";

    if (rel_err < 1e-3) {
      std::cout << "  Result: PASS [K]\n\n";
    } else {
      std::cout << "  Result: FAIL [K]\n\n";
      all_passed = false;
    }
  }

  // -------------------------------------------------------------
  // Test 3: Adjoint Double Layer Potential (K': p0 -> p1)
  // -------------------------------------------------------------
  {
    std::cout << "--- Testing Operator 3: Adjoint Double Layer Potential (K', p0 -> p1) ---\n";
    uniform_spacetime_be_matrix_onthefly_cpu<
      spacetime_heat_adl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p1>,
      uniform_spacetime_be_space<basis_tri_p0>> Kt_cpu(kernel_adl, space_p1, space_p0, 4, 4);

    uniform_spacetime_be_matrix_onthefly_gpu<
      spacetime_heat_adl_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p1>,
      uniform_spacetime_be_space<basis_tri_p0>> Kt_gpu(kernel_adl, space_p1, space_p0, st_mesh_gpu, 4, 4, 2);

    lo dim_dom = Kt_cpu.get_dim_domain();
    lo dim_rng = Kt_cpu.get_dim_range();
    block_vector x(n_timesteps, dim_dom);
    for (lo b = 0; b < n_timesteps; ++b) {
      for (lo i = 0; i < dim_dom; ++i) {
        x.get_block(b).set(i, std::sin((b + 1) * 0.3 + i * 0.5));
      }
    }

    block_vector y_cpu(n_timesteps, dim_rng);
    block_vector y_gpu(n_timesteps, dim_rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    Kt_cpu.apply(x, y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    Kt_gpu.apply(x, y_gpu);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t1 - t0).count();

    block_vector diff(y_gpu);
    diff.add(y_cpu, -1.0);
    sc norm_cpu = y_cpu.norm();
    sc rel_err = diff.norm() / (norm_cpu > 1e-12 ? norm_cpu : 1.0);

    std::cout << "  CPU time:   " << std::fixed << std::setprecision(5) << cpu_time << " s\n";
    std::cout << "  Metal time: " << std::fixed << std::setprecision(5) << gpu_time << " s\n";
    std::cout << "  Speedup:    " << (cpu_time / (gpu_time > 0 ? gpu_time : 1e-6)) << "x\n";
    std::cout << "  ||y_metal - y_cpu|| / ||y_cpu||: " << std::scientific << rel_err << "\n";

    if (rel_err < 1e-3) {
      std::cout << "  Result: PASS [K']\n\n";
    } else {
      std::cout << "  Result: FAIL [K']\n\n";
      all_passed = false;
    }
  }

  // -------------------------------------------------------------
  // Test 4: Hypersingular Potential (D: p1 -> p1)
  // -------------------------------------------------------------
  {
    std::cout << "--- Testing Operator 4: Hypersingular Potential (D, p1 -> p1) ---\n";
    uniform_spacetime_be_matrix_onthefly_cpu<
      spacetime_heat_hs_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p1>,
      uniform_spacetime_be_space<basis_tri_p1>> D_cpu(kernel_hs, space_p1, space_p1, 4, 4);

    uniform_spacetime_be_matrix_onthefly_gpu<
      spacetime_heat_hs_kernel_antiderivative,
      uniform_spacetime_be_space<basis_tri_p1>,
      uniform_spacetime_be_space<basis_tri_p1>> D_gpu(kernel_hs, space_p1, space_p1, st_mesh_gpu, 4, 4, 2);

    lo dim_dom = D_cpu.get_dim_domain();
    lo dim_rng = D_cpu.get_dim_range();
    block_vector x(n_timesteps, dim_dom);
    for (lo b = 0; b < n_timesteps; ++b) {
      for (lo i = 0; i < dim_dom; ++i) {
        x.get_block(b).set(i, std::cos((b + 1) * 0.2 + i * 0.6));
      }
    }

    block_vector y_cpu(n_timesteps, dim_rng);
    block_vector y_gpu(n_timesteps, dim_rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    D_cpu.apply(x, y_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    D_gpu.apply(x, y_gpu);
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t1 - t0).count();

    block_vector diff(y_gpu);
    diff.add(y_cpu, -1.0);
    sc norm_cpu = y_cpu.norm();
    sc rel_err = diff.norm() / (norm_cpu > 1e-12 ? norm_cpu : 1.0);

    std::cout << "  CPU time:   " << std::fixed << std::setprecision(5) << cpu_time << " s\n";
    std::cout << "  Metal time: " << std::fixed << std::setprecision(5) << gpu_time << " s\n";
    std::cout << "  Speedup:    " << (cpu_time / (gpu_time > 0 ? gpu_time : 1e-6)) << "x\n";
    std::cout << "  ||y_metal - y_cpu|| / ||y_cpu||: " << std::scientific << rel_err << "\n";

    if (rel_err < 1e-3) {
      std::cout << "  Result: PASS [D]\n\n";
    } else {
      std::cout << "  Result: FAIL [D]\n\n";
      all_passed = false;
    }
  }

  std::cout << "=========================================================\n";
  if (all_passed) {
    std::cout << "  ALL 4 METAL GPU BEM OPERATORS PASSED VERIFICATION!\n";
  } else {
    std::cout << "  SOME OPERATORS FAILED VERIFICATION.\n";
  }
  std::cout << "=========================================================\n";

  return all_passed ? 0 : 1;
}
