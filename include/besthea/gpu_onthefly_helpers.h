#ifndef INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_
#define INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_

#include "besthea/settings.h"

#include <cmath>
#include <iostream>
#include <vector>

#if defined(BESTHEA_USE_CUDA) && !defined(BESTHEA_USE_METAL)
#include <cuda_runtime.h>
#endif

namespace besthea::bem::onthefly::helpers {
  template< int quadr_order >
  struct quadrature_reference_raw;

  template< int quadr_order >
  struct quadrature_reference_raw_float;

  template< int quadr_order >
  struct quadrature_nodes_raw;

  struct heat_kernel_parameters;
  struct heat_kernel_parameters_float;

  struct gpu_apply_vectors_data;

  class apply_load_distribution;

  struct timer_collection;

  struct gpu_threads_per_block;

  /*!
   * Translates quadrature order to quadrature size -- number of quadrature
   * nodes.
   */
#ifdef __NVCC__
  __host__ __device__
#endif
    constexpr int
    qo2qs( int quadr_order ) {
    // this needs to be changed too if quadrature rules are changed in future
    switch ( quadr_order ) {
      case 5:
        return 49;
      case 4:
        return 36;
      case 2:
        return 9;
      case 1:
      default:
        return 1;
    }
  }

  extern bool is_gpu_quadr_order5_initialized;
  extern bool is_gpu_quadr_order4_initialized;
  extern bool is_gpu_quadr_order2_initialized;
  extern bool is_gpu_quadr_order1_initialized;

}

/*!
 *  Struct containing reference quadrature nodes and quadrature weights as raw
 * data.
 */
template< int quadr_order >
struct besthea::bem::onthefly::helpers::quadrature_reference_raw {
  sc _x1_ref[ qo2qs( quadr_order ) ];
  sc _x2_ref[ qo2qs( quadr_order ) ];
  sc _y1_ref[ qo2qs( quadr_order ) ];
  sc _y2_ref[ qo2qs( quadr_order ) ];
  sc _w[ qo2qs( quadr_order ) ];
};

/*!
 *  Struct containing reference quadrature nodes as float for Metal shaders.
 */
template< int quadr_order >
struct besthea::bem::onthefly::helpers::quadrature_reference_raw_float {
  float _x1_ref[ qo2qs( quadr_order ) ];
  float _x2_ref[ qo2qs( quadr_order ) ];
  float _y1_ref[ qo2qs( quadr_order ) ];
  float _y2_ref[ qo2qs( quadr_order ) ];
  float _w[ qo2qs( quadr_order ) ];
};

/*!
 *  Struct containing mapped quadrature nodes as raw data.
 */
template< int quadr_order >
struct besthea::bem::onthefly::helpers::quadrature_nodes_raw {
  sc xs[ qo2qs( quadr_order ) ];
  sc ys[ qo2qs( quadr_order ) ];
  sc zs[ qo2qs( quadr_order ) ];
};

/*!
 *  Struct containing parameters of heat kernel and other auxiliary variables.
 */
struct besthea::bem::onthefly::helpers::heat_kernel_parameters {
  sc alpha;
  sc sqrt_alpha;
  sc alpha_2;
  sc pi;
  sc sqrt_pi;
  heat_kernel_parameters( sc alpha_ ) {
    this->alpha = alpha_;
    sqrt_alpha = std::sqrt( alpha_ );
    alpha_2 = alpha_ * alpha_;
    pi = M_PI;
    sqrt_pi = std::sqrt( M_PI );
  }
};

/*!
 *  Struct containing parameters of heat kernel as float for Metal shaders.
 */
struct besthea::bem::onthefly::helpers::heat_kernel_parameters_float {
  float alpha;
  float sqrt_alpha;
  float alpha_2;
  float pi;
  float sqrt_pi;
  heat_kernel_parameters_float( float alpha_ ) {
    this->alpha = alpha_;
    sqrt_alpha = std::sqrt( alpha_ );
    alpha_2 = alpha_ * alpha_;
    pi = (float) M_PI;
    sqrt_pi = std::sqrt( (float) M_PI );
  }
};

#if defined(BESTHEA_USE_CUDA) && !defined(BESTHEA_USE_METAL)
#define CUDA_CHECK( err )                                                  \
  if ( err != cudaSuccess ) {                                              \
    std::cerr << "CUDA error " << err << " '" << cudaGetErrorString( err ) \
              << "':\n"                                                    \
              << "  in file '" << __FILE__ << "'\n"                        \
              << "  in function '" << __func__ << "'\n"                    \
              << "  on line " << __LINE__ << "'\n";                        \
    throw std::runtime_error( "BESTHEA Exception: cuda error" );           \
  }
#else
#define CUDA_CHECK( cond ) (void)(cond)
#endif

#define METAL_CHECK( cond, msg )                                           \
  if ( !(cond) ) {                                                         \
    std::cerr << "Metal error: " << msg << "\n"                            \
              << "  in file '" << __FILE__ << "'\n"                        \
              << "  in function '" << __func__ << "'\n"                    \
              << "  on line " << __LINE__ << "'\n";                        \
    throw std::runtime_error( std::string("BESTHEA Metal Exception: ") + msg ); \
  }

#endif /* INCLUDE_BESTHEA_GPU_ONTHEFLY_HELPERS_H_ */
