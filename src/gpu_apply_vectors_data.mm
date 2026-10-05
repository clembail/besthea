#ifdef BESTHEA_USE_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
#include <cuda_runtime.h>
#endif

#include "besthea/gpu_apply_vectors_data.h"
#include "besthea/gpu_onthefly_helpers.h"

besthea::bem::onthefly::gpu_apply_vectors_data::gpu_apply_vectors_data( )
  : h_x( nullptr ) {
}

besthea::bem::onthefly::gpu_apply_vectors_data::~gpu_apply_vectors_data( ) {
  free( );
}

void besthea::bem::onthefly::gpu_apply_vectors_data::allocate( int n_gpus,
  lo x_block_count, lo x_size_of_block, lo y_block_count, lo y_size_of_block ) {
  h_y.resize( n_gpus, nullptr );
  d_x.resize( n_gpus, nullptr );
  d_y.resize( n_gpus, nullptr );
  d_x_float.resize( n_gpus, nullptr );
  d_y_float.resize( n_gpus, nullptr );
  mtl_buffer_x.resize( n_gpus, nullptr );
  mtl_buffer_y.resize( n_gpus, nullptr );
  pitch_x.resize( n_gpus, 0 );
  pitch_y.resize( n_gpus, 0 );
  ld_x.resize( n_gpus, 0 );
  ld_y.resize( n_gpus, 0 );

#ifdef BESTHEA_USE_METAL
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    METAL_CHECK( device != nil, "No Metal device found in gpu_apply_vectors_data::allocate" );

    h_x = new sc[ x_block_count * x_size_of_block ]();

    for ( int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++ ) {
      h_y[ gpu_idx ] = new sc[ y_block_count * y_size_of_block ]();

      size_t bytes_x = x_block_count * x_size_of_block * sizeof( float );
      size_t bytes_y = y_block_count * y_size_of_block * sizeof( float );

      id<MTLBuffer> buf_x = [device newBufferWithLength:std::max(bytes_x, (size_t)64)
                                                options:MTLResourceStorageModeShared];
      id<MTLBuffer> buf_y = [device newBufferWithLength:std::max(bytes_y, (size_t)64)
                                                options:MTLResourceStorageModeShared];
      METAL_CHECK( buf_x != nil && buf_y != nil, "Failed to allocate Metal vector buffers" );

      mtl_buffer_x[ gpu_idx ] = (__bridge_retained void*)buf_x;
      mtl_buffer_y[ gpu_idx ] = (__bridge_retained void*)buf_y;

      d_x_float[ gpu_idx ] = (float*)[buf_x contents];
      d_y_float[ gpu_idx ] = (float*)[buf_y contents];
      d_x[ gpu_idx ] = reinterpret_cast< sc * >( d_x_float[ gpu_idx ] );
      d_y[ gpu_idx ] = reinterpret_cast< sc * >( d_y_float[ gpu_idx ] );

      pitch_x[ gpu_idx ] = x_size_of_block * sizeof( float );
      pitch_y[ gpu_idx ] = y_size_of_block * sizeof( float );
      ld_x[ gpu_idx ] = x_size_of_block;
      ld_y[ gpu_idx ] = y_size_of_block;
    }
  }
#else
  CUDA_CHECK(
    cudaMallocHost( &h_x, x_block_count * x_size_of_block * sizeof( sc ) ) );

  for ( int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++ ) {
    CUDA_CHECK( cudaSetDevice( gpu_idx ) );

    CUDA_CHECK( cudaMallocHost(
      &h_y[ gpu_idx ], y_block_count * y_size_of_block * sizeof( sc ) ) );

    CUDA_CHECK( cudaMallocPitch( &d_x[ gpu_idx ], &pitch_x[ gpu_idx ],
      x_size_of_block * sizeof( sc ), x_block_count ) );
    CUDA_CHECK( cudaMallocPitch( &d_y[ gpu_idx ], &pitch_y[ gpu_idx ],
      y_size_of_block * sizeof( sc ), y_block_count ) );

    ld_x[ gpu_idx ] = pitch_x[ gpu_idx ] / sizeof( sc );
    ld_y[ gpu_idx ] = pitch_y[ gpu_idx ] / sizeof( sc );
  }
#endif
}

void besthea::bem::onthefly::gpu_apply_vectors_data::free( ) {
#ifdef BESTHEA_USE_METAL
  if ( h_x != nullptr ) {
    delete[] h_x;
    h_x = nullptr;
  }
  for ( unsigned int i = 0; i < h_y.size( ); i++ ) {
    if ( h_y[ i ] != nullptr ) {
      delete[] h_y[ i ];
      h_y[ i ] = nullptr;
    }
    if ( mtl_buffer_x[ i ] != nullptr ) {
      id<MTLBuffer> buf_x = (__bridge_transfer id<MTLBuffer>)mtl_buffer_x[ i ];
      (void)buf_x;
      buf_x = nil;
      mtl_buffer_x[ i ] = nullptr;
    }
    if ( mtl_buffer_y[ i ] != nullptr ) {
      id<MTLBuffer> buf_y = (__bridge_transfer id<MTLBuffer>)mtl_buffer_y[ i ];
      (void)buf_y;
      buf_y = nil;
      mtl_buffer_y[ i ] = nullptr;
    }
  }
#else
  if ( h_x != nullptr ) {
    CUDA_CHECK( cudaFreeHost( h_x ) );
  }
  for ( unsigned int i = 0; i < h_y.size( ); i++ ) {
    CUDA_CHECK( cudaSetDevice( i ) );

    CUDA_CHECK( cudaFreeHost( h_y[ i ] ) );
    CUDA_CHECK( cudaFree( d_x[ i ] ) );
    CUDA_CHECK( cudaFree( d_y[ i ] ) );
  }
#endif

  h_x = nullptr;
  h_y.clear( );
  d_x.clear( );
  d_y.clear( );
  d_x_float.clear( );
  d_y_float.clear( );
  mtl_buffer_x.clear( );
  mtl_buffer_y.clear( );
  pitch_x.clear( );
  pitch_y.clear( );
  ld_x.clear( );
  ld_y.clear( );
}
