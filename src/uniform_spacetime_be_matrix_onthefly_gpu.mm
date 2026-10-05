/*
Copyright (c) 2020, VSB - Technical University of Ostrava and Graz University of
Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the names of VSB - Technical University of  Ostrava and Graz
  University of Technology nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific prior
  written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL VSB - TECHNICAL UNIVERSITY OF OSTRAVA AND
GRAZ UNIVERSITY OF TECHNOLOGY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#ifdef xdouble
#undef xdouble
#endif

#include "besthea/basis_tri_p0.h"
#include "besthea/basis_tri_p1.h"
#include "besthea/gpu_onthefly_helpers.h"
#include "besthea/quadrature.h"
#include "besthea/spacetime_heat_adl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_dl_kernel_antiderivative.h"
#include "besthea/spacetime_heat_hs_kernel_antiderivative.h"
#include "besthea/spacetime_heat_sl_kernel_antiderivative.h"
#include "besthea/uniform_spacetime_be_matrix_onthefly_gpu.h"
#include "besthea/uniform_spacetime_be_space.h"

#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <mach-o/dyld.h>
#include <mutex>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace besthea::mesh;
using namespace besthea::bem;
using namespace besthea::bem::onthefly;
using namespace besthea::bem::onthefly::helpers;

// Quadrature reference instances for Metal
static quadrature_reference_raw_float<5> g_metal_quadr_ref_5;
static quadrature_reference_raw_float<4> g_metal_quadr_ref_4;
static quadrature_reference_raw_float<2> g_metal_quadr_ref_2;
static quadrature_reference_raw_float<1> g_metal_quadr_ref_1;

// Metadata struct matching Metal shader layout
struct mesh_raw_metadata_float {
  float timestep;
  long n_temporal_elements;
  long n_elems;
  long n_nodes;
};

// ----------------------------------------------------------------------------
// Metal Context & Pipeline Management
// ----------------------------------------------------------------------------

namespace {

class MetalContext {
 public:
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::mutex mutex;

  static MetalContext & get() {
    static MetalContext instance;
    return instance;
  }

  MetalContext()
    : device(nil), command_queue(nil), library(nil) {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      METAL_CHECK(device != nil, "MetalContext: MTLCreateSystemDefaultDevice failed");

      command_queue = [device newCommandQueue];
      METAL_CHECK(command_queue != nil, "MetalContext: newCommandQueue failed");

      load_library();
    }
  }

  void load_library() {
    NSError * error = nil;

    // 1. Try loading metallib from bundle or executable directory
    char exec_path[1024];
    uint32_t size = sizeof(exec_path);
    std::string bin_dir = ".";
    if (_NSGetExecutablePath(exec_path, &size) == 0) {
      std::string full_path(exec_path);
      size_t last_slash = full_path.find_last_of('/');
      if (last_slash != std::string::npos) {
        bin_dir = full_path.substr(0, last_slash);
      }
    }

    std::vector<std::string> search_paths = {
      bin_dir + "/besthea_shaders.metallib",
      bin_dir + "/../lib/besthea/besthea_shaders.metallib",
      "./besthea_shaders.metallib",
      "besthea_shaders.metallib"
    };

    const char * env_path = std::getenv("BESTHEA_METALLIB_PATH");
    if (env_path != nullptr) {
      search_paths.insert(search_paths.begin(), std::string(env_path));
    }

    for (const auto & path : search_paths) {
      NSString * ns_path = [NSString stringWithUTF8String:path.c_str()];
      if ([[NSFileManager defaultManager] fileExistsAtPath:ns_path]) {
        NSURL * url = [NSURL fileURLWithPath:ns_path];
        library = [device newLibraryWithURL:url error:&error];
        if (library != nil) {
          if (besthea::settings::output_verbosity.warnings >= 1) {
            std::cout << "BESTHEA Info: Loaded Metal library from " << path << std::endl;
          }
          return;
        }
      }
    }

    // 2. Try default library
    library = [device newDefaultLibrary];
    if (library != nil) {
      return;
    }

    // 3. Fallback: compile shader source from file if found
    std::vector<std::string> src_search_paths = {
      bin_dir + "/../src/uniform_spacetime_be_matrix_onthefly_metal.metal",
      "src/uniform_spacetime_be_matrix_onthefly_metal.metal",
      "../src/uniform_spacetime_be_matrix_onthefly_metal.metal"
    };

    for (const auto & src_path : src_search_paths) {
      NSString * ns_src_path = [NSString stringWithUTF8String:src_path.c_str()];
      if ([[NSFileManager defaultManager] fileExistsAtPath:ns_src_path]) {
        NSString * source_code = [NSString stringWithContentsOfFile:ns_src_path
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
        if (source_code != nil) {
          MTLCompileOptions * options = [[MTLCompileOptions alloc] init];
          options.languageVersion = MTLLanguageVersion3_0;
          library = [device newLibraryWithSource:source_code options:options error:&error];
          if (library != nil) {
            if (besthea::settings::output_verbosity.warnings >= 1) {
              std::cout << "BESTHEA Info: Compiled Metal shaders from source " << src_path << std::endl;
            }
            return;
          }
        }
      }
    }

    METAL_CHECK(library != nil, "MetalContext: Failed to load besthea_shaders.metallib");
  }

  id<MTLComputePipelineState> get_pipeline(const std::string & kernel_name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = pipelines.find(kernel_name);
    if (it != pipelines.end()) {
      return it->second;
    }

    @autoreleasepool {
      NSString * fn_name = [NSString stringWithUTF8String:kernel_name.c_str()];
      id<MTLFunction> fn = [library newFunctionWithName:fn_name];
      if (fn == nil) {
        std::cerr << "BESTHEA Error: Metal kernel not found: " << kernel_name << std::endl;
        throw std::runtime_error("Metal kernel not found: " + kernel_name);
      }

      NSError * error = nil;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
      if (pso == nil) {
        std::string err_desc = error ? [[error localizedDescription] UTF8String] : "Unknown";
        std::cerr << "BESTHEA Error: Failed to create pipeline for " << kernel_name << ": " << err_desc << std::endl;
        throw std::runtime_error("Metal pipeline creation failed: " + kernel_name);
      }

      pipelines[kernel_name] = pso;
      return pso;
    }
  }
};

template<class kernel_type, class test_space_type, class trial_space_type>
struct metal_kernel_selector;

template<>
struct metal_kernel_selector<
  spacetime_heat_sl_kernel_antiderivative,
  uniform_spacetime_be_space<basis_tri_p0>,
  uniform_spacetime_be_space<basis_tri_p0>> {
  static std::string get_name(int qo) {
    return "k_apply_ver2_sl_p0_p0_qo" + std::to_string(qo);
  }
  static int get_tpb(int qo) {
    return (qo == 1) ? 64 : 128;
  }
};

template<>
struct metal_kernel_selector<
  spacetime_heat_dl_kernel_antiderivative,
  uniform_spacetime_be_space<basis_tri_p0>,
  uniform_spacetime_be_space<basis_tri_p1>> {
  static std::string get_name(int qo) {
    return "k_apply_ver2_dl_p0_p1_qo" + std::to_string(qo);
  }
  static int get_tpb(int /*qo*/) {
    return 128;
  }
};

template<>
struct metal_kernel_selector<
  spacetime_heat_adl_kernel_antiderivative,
  uniform_spacetime_be_space<basis_tri_p1>,
  uniform_spacetime_be_space<basis_tri_p0>> {
  static std::string get_name(int qo) {
    return "k_apply_ver2_adl_p1_p0_qo" + std::to_string(qo);
  }
  static int get_tpb(int qo) {
    return (qo == 1) ? 64 : 128;
  }
};

template<>
struct metal_kernel_selector<
  spacetime_heat_hs_kernel_antiderivative,
  uniform_spacetime_be_space<basis_tri_p1>,
  uniform_spacetime_be_space<basis_tri_p1>> {
  static std::string get_name(int qo) {
    return "k_apply_ver2_hs_p1_p1_qo" + std::to_string(qo);
  }
  static int get_tpb(int /*qo*/) {
    return 64;
  }
};

}  // anonymous namespace

// ----------------------------------------------------------------------------
// uniform_spacetime_be_matrix_onthefly_gpu Implementation
// ----------------------------------------------------------------------------

template<class kernel_type, class test_space_type, class trial_space_type>
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
uniform_spacetime_be_matrix_onthefly_gpu(kernel_type & kernel,
  test_space_type & test_space, trial_space_type & trial_space,
  const uniform_spacetime_tensor_mesh_gpu & gpu_mesh_,
  int order_singular, int order_regular, int gpu_kernel_version_,
  bool loadbalancing_use_cpu_)
  : uniform_spacetime_be_matrix_onthefly_cpu<
      kernel_type, test_space_type, trial_space_type>(
        kernel, test_space, trial_space, order_singular, order_regular),
    gpu_mesh(&gpu_mesh_),
    n_gpus(gpu_mesh_.get_n_gpus()),
    gpu_kernel_version(gpu_kernel_version_),
    load_distr(nullptr),
    loadbalancing_use_cpu(loadbalancing_use_cpu_) {

  if (n_gpus == 0) {
    if (besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: Trying to use gpu version of onthefly "
                   "matrix class, but no Metal-capable devices were found. "
                   "Using cpu-only version.\n";
    }
  }

  if (gpu_kernel_version != 2) {
    if (besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: gpu_kernel_version=" << gpu_kernel_version
                << " requested. Defaulting to optimized Metal version 2.\n";
    }
    this->gpu_kernel_version = 2;
  }

  if (n_gpus > 0) {
    init_gpu_data();
  }
}

template<class kernel_type, class test_space_type, class trial_space_type>
besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
~uniform_spacetime_be_matrix_onthefly_gpu() {
  delete this->load_distr;
}

template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
init_gpu_data() {
  // Version 2: x_perm, y
  vectors_data.allocate(n_gpus,
    this->get_dim_domain(), this->get_block_dim(),
    this->get_block_dim(), this->get_dim_range());

  lo gpu_chunk_size = metal_kernel_selector<kernel_type, test_space_type, trial_space_type>::get_tpb(this->_order_regular);

  this->load_distr = new gpu_apply_load_distribution(
    n_gpus, gpu_mesh->get_metadata().n_elems,
    gpu_chunk_size, loadbalancing_use_cpu);

  switch (this->_order_regular) {
    case 5:
      if (!is_gpu_quadr_order5_initialized) {
        init_gpu_quadrature_memory<5>();
        is_gpu_quadr_order5_initialized = true;
      }
      break;
    case 4:
      if (!is_gpu_quadr_order4_initialized) {
        init_gpu_quadrature_memory<4>();
        is_gpu_quadr_order4_initialized = true;
      }
      break;
    case 2:
      if (!is_gpu_quadr_order2_initialized) {
        init_gpu_quadrature_memory<2>();
        is_gpu_quadr_order2_initialized = true;
      }
      break;
    case 1:
    default:
      if (!is_gpu_quadr_order1_initialized) {
        init_gpu_quadrature_memory<1>();
        is_gpu_quadr_order1_initialized = true;
      }
      break;
  }
}

template<class kernel_type, class test_space_type, class trial_space_type>
template<int quadr_order>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
init_gpu_quadrature_memory() const {
  constexpr int qs = qo2qs(quadr_order);
  quadrature_reference_raw_float<quadr_order> quadr_ref_tmp;
  for (int i = 0; i < qs; ++i) {
    quadr_ref_tmp._x1_ref[i] = (float)this->quadr_reference._x1_ref[0][i];
    quadr_ref_tmp._x2_ref[i] = (float)this->quadr_reference._x2_ref[0][i];
    quadr_ref_tmp._y1_ref[i] = (float)this->quadr_reference._y1_ref[0][i];
    quadr_ref_tmp._y2_ref[i] = (float)this->quadr_reference._y2_ref[0][i];
    quadr_ref_tmp._w[i]      = (float)this->quadr_reference._w[0][i];
  }

  if constexpr (quadr_order == 5) {
    g_metal_quadr_ref_5 = quadr_ref_tmp;
  } else if constexpr (quadr_order == 4) {
    g_metal_quadr_ref_4 = quadr_ref_tmp;
  } else if constexpr (quadr_order == 2) {
    g_metal_quadr_ref_2 = quadr_ref_tmp;
  } else {
    g_metal_quadr_ref_1 = quadr_ref_tmp;
  }
}

template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
apply(const block_vector_type & x, block_vector_type & y, bool trans, sc alpha, sc beta) const {
  if (n_gpus == 0) {
    uniform_spacetime_be_matrix_onthefly_cpu<
      kernel_type, test_space_type, trial_space_type>::
      apply(x, y, trans, alpha, beta);
    return;
  }

  if (trans) {
    if (besthea::settings::output_verbosity.warnings >= 1) {
      std::cerr << "BESTHEA Warning: transposed onthefly matrices are not "
                   "supported. Apply operation will not be performed.\n";
    }
    return;
  }

  if (x.get_n_blocks() != this->get_block_dim() ||
      x.get_size_of_block() != this->get_dim_domain()) {
    std::cerr << "BESTHEA Error: x block vector dimension ("
              << x.get_n_blocks() << "*" << x.get_size_of_block()
              << ") does not match block matrix domain dimension ("
              << this->get_block_dim() << "*" << this->get_dim_domain()
              << ").\n";
    throw std::runtime_error("BESTHEA Exception: incompatible matrix and vector dimensions");
  }

  if (y.get_n_blocks() != this->get_block_dim() ||
      y.get_size_of_block() != this->get_dim_range()) {
    std::cerr << "BESTHEA Error: y block vector dimension ("
              << y.get_n_blocks() << "*" << y.get_size_of_block()
              << ") does not match block matrix range dimension ("
              << this->get_block_dim() << "*" << this->get_dim_range()
              << "). Apply will not be performed.\n";
    throw std::runtime_error("BESTHEA Exception: incompatible matrix and vector dimensions");
  }

  gpu_apply_timer_collection timers(n_gpus);
  timers.combined.start();

  block_vector_type y_perm;
  block_vector_type x_perm;

  timers.cpu_scalein.start();
  x_perm.copy_permute(x);
  y_perm.resize_to_match_permute(y, true);
  y.scale(beta);
  timers.cpu_scalein.stop();

  if (besthea::settings::output_verbosity.onthefly_loadbalance >= 1) {
    load_distr->print();
  }

  if (load_distr->get_gpu_count_total() > 0) {
    this->apply_gpu_treg_sreg_begin(x_perm, y, alpha, timers);
  }

  timers.cpu_all.start();
  if (load_distr->get_cpu_count() > 0) {
    timers.cpu_treg_sreg.start();
    this->apply_cpu_treg_sreg(x_perm, y_perm, alpha,
      load_distr->get_cpu_begin(), load_distr->get_cpu_end());
    timers.cpu_treg_sreg.stop();
  }
  timers.cpu_treg_ssng.start();
  this->apply_cpu_treg_ssng(x_perm, y_perm, alpha);
  timers.cpu_treg_ssng.stop();
  timers.cpu_tsng.start();
  this->apply_cpu_tsng(x_perm, y_perm, alpha);
  timers.cpu_tsng.stop();
  timers.cpu_all.stop();

  if (load_distr->get_gpu_count_total() > 0) {
    this->apply_gpu_treg_sreg_finalize(y);
  }

  y.add_permute(y_perm);

  timers.combined.stop();

  if (besthea::settings::output_verbosity.timers >= 1) {
    if (besthea::settings::output_verbosity.timers >= 2) {
      timers.print_all();
    }
    std::cout << "BESTHEA Info: apply elapsed time = "
              << timers.combined.get_elapsed_time_in_seconds() << " seconds\n";
  }

  load_distr->adapt(
    timers.get_cpu_time_const(),
    timers.get_cpu_time_scaling(),
    timers.get_gpu_time_const(),
    timers.get_gpu_time_scaling());
}

template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
apply_gpu_treg_sreg_begin(const block_vector_type & x,
  const block_vector_type & y, sc alpha,
  gpu_apply_timer_collection & timers) const {

  x.copy_to_raw(vectors_data.h_x);

  size_t total_x = x.get_n_blocks() * x.get_size_of_block();
  size_t total_y = y.get_n_blocks() * y.get_size_of_block();

  mtl_command_buffers.clear();
  mtl_command_buffers.resize(n_gpus, nullptr);

  auto & metal_ctx = MetalContext::get();

  for (int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    timers.gpu_all[gpu_idx].start_submit();
    timers.gpu_copyin[gpu_idx].start_submit();

    // Copy double h_x to float d_x_float buffer
    float * dx_f = vectors_data.d_x_float[gpu_idx];
    for (size_t i = 0; i < total_x; ++i) {
      dx_f[i] = (float)vectors_data.h_x[i];
    }
    // Zero out d_y_float buffer
    float * dy_f = vectors_data.d_y_float[gpu_idx];
    std::memset(dy_f, 0, total_y * sizeof(float));

    timers.gpu_copyin[gpu_idx].stop_submit();
  }

  const auto & mesh_meta = gpu_mesh->get_metadata();
  mesh_raw_metadata_float meta_f;
  meta_f.timestep = (float)mesh_meta.timestep;
  meta_f.n_temporal_elements = mesh_meta.n_temporal_elements;
  meta_f.n_elems = mesh_meta.n_elems;
  meta_f.n_nodes = mesh_meta.n_nodes;

  heat_kernel_parameters_float kp((float)this->_kernel->get_alpha());
  float alpha_f = (float)alpha;

  std::string kernel_name = metal_kernel_selector<kernel_type, test_space_type, trial_space_type>::get_name(this->_order_regular);
  int tpbx = metal_kernel_selector<kernel_type, test_space_type, trial_space_type>::get_tpb(this->_order_regular);
  id<MTLComputePipelineState> pso = metal_ctx.get_pipeline(kernel_name);

  for (int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    lo gpu_tst_elem_begin = load_distr->get_gpu_begin(gpu_idx);
    lo gpu_tst_elem_count = load_distr->get_gpu_count(gpu_idx);

    if (gpu_tst_elem_count == 0) {
      continue;
    }

    timers.gpu_compute[gpu_idx].start_submit();

    @autoreleasepool {
      id<MTLCommandBuffer> cmd_buffer = [metal_ctx.command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

      [encoder setComputePipelineState:pso];

      id<MTLBuffer> buf_x = (__bridge id<MTLBuffer>)vectors_data.mtl_buffer_x[gpu_idx];
      id<MTLBuffer> buf_y = (__bridge id<MTLBuffer>)vectors_data.mtl_buffer_y[gpu_idx];
      [encoder setBuffer:buf_x offset:0 atIndex:0];
      [encoder setBuffer:buf_y offset:0 atIndex:1];

      long ld_x_val = vectors_data.ld_x[gpu_idx];
      long ld_y_val = vectors_data.ld_y[gpu_idx];
      [encoder setBytes:&ld_x_val length:sizeof(long) atIndex:2];
      [encoder setBytes:&ld_y_val length:sizeof(long) atIndex:3];
      [encoder setBytes:&alpha_f length:sizeof(float) atIndex:4];

      long i_tst_begin_val = gpu_tst_elem_begin;
      [encoder setBytes:&i_tst_begin_val length:sizeof(long) atIndex:5];
      [encoder setBytes:&meta_f length:sizeof(meta_f) atIndex:6];

      const auto & per_gpu_mesh = gpu_mesh->get_per_gpu_data()[gpu_idx];
      id<MTLBuffer> buf_areas = (__bridge id<MTLBuffer>)per_gpu_mesh.mtl_buf_areas;
      id<MTLBuffer> buf_coords = (__bridge id<MTLBuffer>)per_gpu_mesh.mtl_buf_coords;
      id<MTLBuffer> buf_nodes = (__bridge id<MTLBuffer>)per_gpu_mesh.mtl_buf_nodes;
      id<MTLBuffer> buf_normals = (__bridge id<MTLBuffer>)per_gpu_mesh.mtl_buf_normals;

      [encoder setBuffer:buf_areas offset:0 atIndex:7];
      [encoder setBuffer:buf_coords offset:0 atIndex:8];
      [encoder setBuffer:buf_nodes offset:0 atIndex:9];
      [encoder setBuffer:buf_normals offset:0 atIndex:10];

      [encoder setBytes:&kp length:sizeof(kp) atIndex:11];

      switch (this->_order_regular) {
        case 5:
          [encoder setBytes:&g_metal_quadr_ref_5 length:sizeof(g_metal_quadr_ref_5) atIndex:12];
          break;
        case 4:
          [encoder setBytes:&g_metal_quadr_ref_4 length:sizeof(g_metal_quadr_ref_4) atIndex:12];
          break;
        case 2:
          [encoder setBytes:&g_metal_quadr_ref_2 length:sizeof(g_metal_quadr_ref_2) atIndex:12];
          break;
        case 1:
        default:
          [encoder setBytes:&g_metal_quadr_ref_1 length:sizeof(g_metal_quadr_ref_1) atIndex:12];
          break;
      }

      MTLSize grid_size = MTLSizeMake(gpu_tst_elem_count, 1, 1);
      MTLSize block_size = MTLSizeMake(tpbx, 1, 1);
      [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:block_size];
      [encoder endEncoding];

      [cmd_buffer commit];
      mtl_command_buffers[gpu_idx] = (__bridge_retained void*)cmd_buffer;
    }
  }
}

template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
  kernel_type, test_space_type, trial_space_type>::
apply_gpu_treg_sreg_finalize(block_vector_type & y) const {
  size_t total_y = y.get_n_blocks() * y.get_size_of_block();

  for (int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
    if (mtl_command_buffers[gpu_idx] != nullptr) {
      @autoreleasepool {
        id<MTLCommandBuffer> cmd_buffer = (__bridge_transfer id<MTLCommandBuffer>)mtl_command_buffers[gpu_idx];
        mtl_command_buffers[gpu_idx] = nullptr;
        [cmd_buffer waitUntilCompleted];
      }
    }

    // Convert float d_y_float back to double h_y
    float * dy_f = vectors_data.d_y_float[gpu_idx];
    sc * hy = vectors_data.h_y[gpu_idx];
    for (size_t i = 0; i < total_y; ++i) {
      hy[i] = (sc)dy_f[i];
    }
  }

#pragma omp parallel for
  for (lo b = 0; b < y.get_n_blocks(); b++) {
    vector_type & y_block = y.get_block(b);
    for (int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++) {
      const sc * h_y_block = vectors_data.h_y[gpu_idx] + b * y.get_size_of_block();
      y_block.add_from_raw(h_y_block, 1.0);
    }
  }
}

// ----------------------------------------------------------------------------
// Explicit Template Instantiations
// ----------------------------------------------------------------------------

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_sl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_dl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_adl_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 > >;

template class
  besthea::bem::onthefly::uniform_spacetime_be_matrix_onthefly_gpu<
    besthea::bem::spacetime_heat_hs_kernel_antiderivative,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 >,
    besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p1 > >;
