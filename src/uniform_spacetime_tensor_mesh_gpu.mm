#ifdef BESTHEA_USE_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
#include <cuda_runtime.h>
#endif

#include "besthea/uniform_spacetime_tensor_mesh_gpu.h"
#include "besthea/gpu_onthefly_helpers.h"
#include <exception>
#include <iostream>

besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  uniform_spacetime_tensor_mesh_gpu(
    const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh ) {
  metadata.timestep = orig_mesh.get_timestep( );
  metadata.n_temporal_elements = orig_mesh.get_n_temporal_elements( );
  metadata.n_elems = orig_mesh.get_spatial_surface_mesh( )->get_n_elements( );
  metadata.n_nodes = orig_mesh.get_spatial_surface_mesh( )->get_n_nodes( );

#ifdef BESTHEA_USE_METAL
  n_gpus = 0;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if ( device == nil ) {
      if ( besthea::settings::output_verbosity.warnings >= 1 ) {
        std::cerr << "BESTHEA Warning: Constructing GPU mesh, "
                     "but no Metal-capable GPU was detected.\n";
      }
      n_gpus = 0;
      return;
    }
    n_gpus = 1;
    per_gpu_data.resize( n_gpus );

    mesh_raw_data & curr_gpu_data = per_gpu_data[ 0 ];

    size_t areas_bytes = metadata.n_elems * sizeof( float );
    size_t coords_bytes = 3 * metadata.n_nodes * sizeof( float );
    size_t nodes_bytes = 3 * metadata.n_elems * sizeof( lo );
    size_t normals_bytes = 3 * metadata.n_elems * sizeof( float );

    id<MTLBuffer> buf_areas = [device newBufferWithLength:std::max(areas_bytes, (size_t)64)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_coords = [device newBufferWithLength:std::max(coords_bytes, (size_t)64)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_nodes = [device newBufferWithLength:std::max(nodes_bytes, (size_t)64)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_normals = [device newBufferWithLength:std::max(normals_bytes, (size_t)64)
                                                    options:MTLResourceStorageModeShared];

    METAL_CHECK( buf_areas != nil && buf_coords != nil && buf_nodes != nil && buf_normals != nil,
                 "Failed to allocate Metal mesh buffers" );

    curr_gpu_data.mtl_buf_areas = (__bridge_retained void*)buf_areas;
    curr_gpu_data.mtl_buf_coords = (__bridge_retained void*)buf_coords;
    curr_gpu_data.mtl_buf_nodes = (__bridge_retained void*)buf_nodes;
    curr_gpu_data.mtl_buf_normals = (__bridge_retained void*)buf_normals;

    curr_gpu_data.d_element_areas_float = (float*)[buf_areas contents];
    curr_gpu_data.d_node_coords_float = (float*)[buf_coords contents];
    curr_gpu_data.d_element_nodes = (lo*)[buf_nodes contents];
    curr_gpu_data.d_element_normals_float = (float*)[buf_normals contents];

    // Compatibility pointers
    curr_gpu_data.d_element_areas = reinterpret_cast< sc * >( curr_gpu_data.d_element_areas_float );
    curr_gpu_data.d_node_coords = reinterpret_cast< sc * >( curr_gpu_data.d_node_coords_float );
    curr_gpu_data.d_element_normals = reinterpret_cast< sc * >( curr_gpu_data.d_element_normals_float );

    // Convert host areas (double) to device float
    const auto & host_areas = orig_mesh.get_spatial_surface_mesh( )->get_areas( );
    for ( lo i = 0; i < metadata.n_elems; ++i ) {
      curr_gpu_data.d_element_areas_float[ i ] = static_cast< float >( host_areas[ i ] );
    }

    // Convert host node coords (double) to device float
    const auto & host_nodes = orig_mesh.get_spatial_surface_mesh( )->get_nodes( );
    for ( lo i = 0; i < 3 * metadata.n_nodes; ++i ) {
      curr_gpu_data.d_node_coords_float[ i ] = static_cast< float >( host_nodes[ i ] );
    }

    // Copy element node indices
    const auto & host_elements = orig_mesh.get_spatial_surface_mesh( )->get_elements( );
    std::copy( host_elements.begin( ), host_elements.end( ), curr_gpu_data.d_element_nodes );

    // Convert host normals (double) to device float
    const auto & host_normals = orig_mesh.get_spatial_surface_mesh( )->get_normals( );
    for ( lo i = 0; i < 3 * metadata.n_elems; ++i ) {
      curr_gpu_data.d_element_normals_float[ i ] = static_cast< float >( host_normals[ i ] );
    }
  }
#else
  n_gpus = 0;
  cudaError_t errDc = cudaGetDeviceCount( &n_gpus );
  if ( errDc != cudaErrorNoDevice )
    CUDA_CHECK( errDc );

  if ( n_gpus == 0 || errDc == cudaErrorNoDevice ) {
    if ( besthea::settings::output_verbosity.warnings >= 1 ) {
      std::cerr << "BESTHEA Warning: Constructing GPU mesh, "
                   "but no cuda-capable GPUs were detected.\n";
    }
    n_gpus = 0;
  }

  per_gpu_data.resize( n_gpus );
  for ( int gpu_idx = 0; gpu_idx < n_gpus; gpu_idx++ ) {
    CUDA_CHECK( cudaSetDevice( gpu_idx ) );

    mesh_raw_data & curr_gpu_data = per_gpu_data[ gpu_idx ];

    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_areas,
      1 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_areas ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_node_coords,
      3 * metadata.n_nodes * sizeof( *curr_gpu_data.d_node_coords ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_nodes,
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_nodes ) ) );
    CUDA_CHECK( cudaMalloc( &curr_gpu_data.d_element_normals,
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_normals ) ) );

    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_areas,
      orig_mesh.get_spatial_surface_mesh( )->get_areas( ).data( ),
      1 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_areas ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_node_coords,
      orig_mesh.get_spatial_surface_mesh( )->get_nodes( ).data( ),
      3 * metadata.n_nodes * sizeof( *curr_gpu_data.d_node_coords ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_nodes,
      orig_mesh.get_spatial_surface_mesh( )->get_elements( ).data( ),
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_nodes ),
      cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( curr_gpu_data.d_element_normals,
      orig_mesh.get_spatial_surface_mesh( )->get_normals( ).data( ),
      3 * metadata.n_elems * sizeof( *curr_gpu_data.d_element_normals ),
      cudaMemcpyHostToDevice ) );
  }
#endif
}

besthea::mesh::uniform_spacetime_tensor_mesh_gpu::
  ~uniform_spacetime_tensor_mesh_gpu( ) {
  free( );
}

void besthea::mesh::uniform_spacetime_tensor_mesh_gpu::free( ) {
#ifdef BESTHEA_USE_METAL
  for ( unsigned int gpu_idx = 0; gpu_idx < per_gpu_data.size( ); gpu_idx++ ) {
    mesh_raw_data & curr_gpu_data = per_gpu_data[ gpu_idx ];
    if ( curr_gpu_data.mtl_buf_areas != nullptr ) {
      id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)curr_gpu_data.mtl_buf_areas;
      (void)b;
      b = nil;
      curr_gpu_data.mtl_buf_areas = nullptr;
    }
    if ( curr_gpu_data.mtl_buf_coords != nullptr ) {
      id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)curr_gpu_data.mtl_buf_coords;
      (void)b;
      b = nil;
      curr_gpu_data.mtl_buf_coords = nullptr;
    }
    if ( curr_gpu_data.mtl_buf_nodes != nullptr ) {
      id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)curr_gpu_data.mtl_buf_nodes;
      (void)b;
      b = nil;
      curr_gpu_data.mtl_buf_nodes = nullptr;
    }
    if ( curr_gpu_data.mtl_buf_normals != nullptr ) {
      id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)curr_gpu_data.mtl_buf_normals;
      (void)b;
      b = nil;
      curr_gpu_data.mtl_buf_normals = nullptr;
    }
  }
#else
  for ( unsigned int gpu_idx = 0; gpu_idx < per_gpu_data.size( ); gpu_idx++ ) {
    CUDA_CHECK( cudaSetDevice( gpu_idx ) );

    mesh_raw_data & curr_gpu_data = per_gpu_data[ gpu_idx ];

    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_areas ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_node_coords ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_nodes ) );
    CUDA_CHECK( cudaFree( curr_gpu_data.d_element_normals ) );
  }
#endif
  per_gpu_data.clear( );
}
