#ifndef INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_
#define INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_

#include "besthea/settings.h"
#include "besthea/uniform_spacetime_tensor_mesh.h"

#include <vector>

namespace besthea::mesh {
  class uniform_spacetime_tensor_mesh_gpu;
}

/*!
 *  Class representing uniform spacetime mesh resident in the GPU memory.
 */
class besthea::mesh::uniform_spacetime_tensor_mesh_gpu {
 public:
  /*!
   *  Struct containing GPU-resident space mesh raw data.
   */
  struct mesh_raw_data {
    sc * d_element_areas;
    sc * d_node_coords;      // XYZXYZXYZXYZ...
    lo * d_element_nodes;    // 123123123123...
    sc * d_element_normals;  // XYZXYZXYZXYZ
    float * d_element_areas_float;
    float * d_node_coords_float;
    float * d_element_normals_float;
    void * mtl_buf_areas;
    void * mtl_buf_coords;
    void * mtl_buf_nodes;
    void * mtl_buf_normals;
    mesh_raw_data( )
      : d_element_areas( nullptr ),
        d_node_coords( nullptr ),
        d_element_nodes( nullptr ),
        d_element_normals( nullptr ),
        d_element_areas_float( nullptr ),
        d_node_coords_float( nullptr ),
        d_element_normals_float( nullptr ),
        mtl_buf_areas( nullptr ),
        mtl_buf_coords( nullptr ),
        mtl_buf_nodes( nullptr ),
        mtl_buf_normals( nullptr ) {
    }
  };

  /*!
   *  Struct containing spacetime mesh metadata.
   */
  struct mesh_raw_metadata {
    sc timestep;
    lo n_temporal_elements;
    lo n_elems;
    lo n_nodes;
  };

 public:
  /*!
   * Constructor. Creates this instance and copies necessary data to GPU memory.
   * @param[in] orig_mesh The original mesh.
   */
  explicit uniform_spacetime_tensor_mesh_gpu(
    const besthea::mesh::uniform_spacetime_tensor_mesh & orig_mesh );

  /*!
   * Destructor.
   */
  ~uniform_spacetime_tensor_mesh_gpu( );

  /*!
   * Returns metadata structure holding information about this mesh.
   */
  const mesh_raw_metadata & get_metadata( ) const {
    return metadata;
  }

  /*!
   * Returns vector of structures holding pointers to data on GPUs.
   */
  const std::vector< mesh_raw_data > & get_per_gpu_data( ) const {
    return per_gpu_data;
  }

  /*!
   * Returns the used number of GPUs.
   */
  int get_n_gpus( ) const {
    return n_gpus;
  }

 private:
  /*!
   * Frees the allocated GPU memory.
   */
  void free( );

 private:
  mesh_raw_metadata metadata;  //!< Metadata about this mesh.
  std::vector< mesh_raw_data >
    per_gpu_data;  //!< Pointers to GPU-resident data.
  int n_gpus;      //!< Number of used GPUs.
};

#endif /* INCLUDE_BESTHEA_UNIFORM_SPACETIME_TENSOR_MESH_GPU_H_ */
