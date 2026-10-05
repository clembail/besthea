#ifndef INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_
#define INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_

#include "besthea/settings.h"

#include <vector>

namespace besthea::bem::onthefly {
  struct gpu_apply_vectors_data;
}

/*!
 *  Struct containing CPU and GPU resident vectors data.
 */
struct besthea::bem::onthefly::gpu_apply_vectors_data {
  /*!
   * Constructor
   */
  gpu_apply_vectors_data( );

  /*!
   * Copy constructor - deleted
   */
  gpu_apply_vectors_data( const gpu_apply_vectors_data & that ) = delete;

  /*!
   * Destructor
   */
  ~gpu_apply_vectors_data( );

  /*!
   * Allocates the data vectors
   * @param[in] n_gpus Number of GPU devices
   * @param[in] x_block_count Number of blocks in vector x
   * @param[in] x_size_of_block Size of block in vector x
   * @param[in] y_block_count Number of blocks in vector y
   * @param[in] y_size_of_block Size of block in vector y
   */
  void allocate( int n_gpus, lo x_block_count, lo x_size_of_block,
    lo y_block_count, lo y_size_of_block );

  /*!
   * Frees the memory of the vectors
   */
  void free( );

  sc * h_x;                                        //<! raw data on host
  std::vector< sc * > h_y;                         //<! raw data on host
  std::vector< float * > d_x_float, d_y_float;     //<! float data on device
  std::vector< void * > mtl_buffer_x, mtl_buffer_y; //<! Metal MTLBuffer handles
  std::vector< sc * > d_x, d_y;                    //<! raw data on device (pointers)
  std::vector< size_t > pitch_x, pitch_y;          //<! pitch in bytes
  std::vector< lo > ld_x, ld_y;                    //<! leading dimension in elements
};

#endif /* INCLUDE_BESTHEA_GPU_APPLY_VECTORS_DATA_H_ */
