#include <metal_stdlib>
using namespace metal;

constexpr int qo2qs(int quadr_order) {
    switch (quadr_order) {
        case 5: return 49;
        case 4: return 36;
        case 2: return 9;
        case 1:
        default: return 1;
    }
}

template<int quadr_order>
struct quadrature_reference_raw_float {
    float _x1_ref[qo2qs(quadr_order)];
    float _x2_ref[qo2qs(quadr_order)];
    float _y1_ref[qo2qs(quadr_order)];
    float _y2_ref[qo2qs(quadr_order)];
    float _w[qo2qs(quadr_order)];
};

template<int quadr_order>
struct quadrature_nodes_raw_float {
    float xs[qo2qs(quadr_order)];
    float ys[qo2qs(quadr_order)];
    float zs[qo2qs(quadr_order)];
};

struct heat_kernel_parameters_float {
    float alpha;
    float sqrt_alpha;
    float alpha_2;
    float pi;
    float sqrt_pi;
};

struct mesh_raw_metadata_float {
    float timestep;
    long n_temporal_elements;
    long n_elems;
    long n_nodes;
};

inline float metal_erf(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float ax = metal::abs(x);
    if (ax > 5.0f) return sign;
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * metal::exp(-ax * ax));
}

template<int quadr_order>
inline void d_triangles_to_geometry_000_tst_shmem(
    long i_tst,
    device const long * d_element_nodes,
    device const float * d_node_coords,
    threadgroup quadrature_nodes_raw_float<quadr_order> & shmem_quadr_nodes_tst,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref,
    uint tid,
    uint block_dim_x)
{
    device const long * tst_elem_nodes = d_element_nodes + 3 * i_tst;

    device const float * x1 = d_node_coords + 3 * tst_elem_nodes[0];
    device const float * x2 = d_node_coords + 3 * tst_elem_nodes[1];
    device const float * x3 = d_node_coords + 3 * tst_elem_nodes[2];

    constant const float * x1_ref = quadr_ref._x1_ref;
    constant const float * x2_ref = quadr_ref._x2_ref;

    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i = (int)tid; i < quadr_size; i += (int)block_dim_x) {
        shmem_quadr_nodes_tst.xs[i] = x1[0]
            + (x2[0] - x1[0]) * x1_ref[i]
            + (x3[0] - x1[0]) * x2_ref[i];
        shmem_quadr_nodes_tst.ys[i] = x1[1]
            + (x2[1] - x1[1]) * x1_ref[i]
            + (x3[1] - x1[1]) * x2_ref[i];
        shmem_quadr_nodes_tst.zs[i] = x1[2]
            + (x2[2] - x1[2]) * x1_ref[i]
            + (x3[2] - x1[2]) * x2_ref[i];
    }
}

template<int quadr_order>
inline void d_triangles_to_geometry_000_trl(
    long i_trl,
    device const long * d_element_nodes,
    device const float * d_node_coords,
    thread quadrature_nodes_raw_float<quadr_order> & quadr_nodes_trl,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref)
{
    device const long * trl_elem_nodes = d_element_nodes + 3 * i_trl;

    device const float * y1 = d_node_coords + 3 * trl_elem_nodes[0];
    device const float * y2 = d_node_coords + 3 * trl_elem_nodes[1];
    device const float * y3 = d_node_coords + 3 * trl_elem_nodes[2];

    constant const float * y1_ref = quadr_ref._y1_ref;
    constant const float * y2_ref = quadr_ref._y2_ref;
    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i = 0; i < quadr_size; ++i) {
        quadr_nodes_trl.xs[i] = y1[0]
            + (y2[0] - y1[0]) * y1_ref[i]
            + (y3[0] - y1[0]) * y2_ref[i];
        quadr_nodes_trl.ys[i] = y1[1]
            + (y2[1] - y1[1]) * y1_ref[i]
            + (y3[1] - y1[1]) * y2_ref[i];
        quadr_nodes_trl.zs[i] = y1[2]
            + (y2[2] - y1[2]) * y1_ref[i]
            + (y3[2] - y1[2]) * y2_ref[i];
    }
}

inline float d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
    float xy1, float xy2, float xy3, float ttau, float ttau_sqrt,
    constant heat_kernel_parameters_float & kp)
{
    constexpr float _two = 2.0f;
    constexpr float _four = 4.0f;
    constexpr float _eight = 8.0f;

    float norm = metal::sqrt(xy1 * xy1 + xy2 * xy2 + xy3 * xy3);
    float sqrt_d = ttau_sqrt;

    float value = (ttau / (_four * kp.pi * kp.alpha * norm)
                   + norm / (_eight * kp.pi * kp.alpha_2))
        * metal_erf(norm / (_two * sqrt_d * kp.sqrt_alpha))
      + sqrt_d / (_four * kp.pi * kp.alpha * kp.sqrt_pi * kp.sqrt_alpha)
        * metal::exp(-(norm * norm) / (_four * ttau * kp.alpha));

    return value;
}

inline float d_dl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
    float xy1, float xy2, float xy3, device const float * ny, float ttau, float ttau_sqrt,
    constant heat_kernel_parameters_float & kp)
{
    constexpr float _one = 1.0f;
    constexpr float _two = 2.0f;
    constexpr float _four = 4.0f;

    float norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
    float norm = metal::sqrt(norm2);
    float dot = xy1 * ny[0] + xy2 * ny[1] + xy3 * ny[2];
    float sqrt_d = ttau_sqrt;

    float value = -dot / (_four * kp.pi * norm)
               * ((_one / (_two * kp.alpha) - ttau / norm2)
               * metal_erf(norm / (_two * sqrt_d * kp.sqrt_alpha))
             + sqrt_d / (kp.sqrt_pi * kp.sqrt_alpha * norm)
               * metal::exp(-norm2 / (_four * kp.alpha * ttau)));

    return value;
}

inline float d_adl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
    float xy1, float xy2, float xy3, device const float * nx, float ttau, float ttau_sqrt,
    constant heat_kernel_parameters_float & kp)
{
    constexpr float _one = 1.0f;
    constexpr float _two = 2.0f;
    constexpr float _four = 4.0f;

    float norm2 = xy1 * xy1 + xy2 * xy2 + xy3 * xy3;
    float norm = metal::sqrt(norm2);
    float dot = -xy1 * nx[0] - xy2 * nx[1] - xy3 * nx[2];
    float sqrt_d = ttau_sqrt;

    float value = -dot / (_four * kp.pi * norm)
               * ((_one / (_two * kp.alpha) - ttau / norm2)
               * metal_erf(norm / (_two * sqrt_d * kp.sqrt_alpha))
             + sqrt_d / (kp.sqrt_pi * kp.sqrt_alpha * norm)
               * metal::exp(-norm2 / (_four * kp.alpha * ttau)));

    return value;
}

inline void d_hs_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
    float xy1, float xy2, float xy3, device const float * nx, device const float * ny,
    float ttau, float ttau_sqrt, thread float * value1, thread float * value2,
    constant heat_kernel_parameters_float & kp)
{
    constexpr float _two = 2.0f;
    constexpr float _four = 4.0f;
    constexpr float _eight = 8.0f;

    float dot = nx[0] * ny[0] + nx[1] * ny[1] + nx[2] * ny[2];
    float norm = metal::sqrt(xy1 * xy1 + xy2 * xy2 + xy3 * xy3);
    float sqrt_d = ttau_sqrt;
    float erf_value = metal_erf(norm / (_two * sqrt_d * kp.sqrt_alpha));
    float four_pi_alpha_norm = _four * kp.pi * kp.alpha * norm;

    *value1 = (ttau / four_pi_alpha_norm + norm / (_eight * kp.pi * kp.alpha_2))
        * erf_value
      + sqrt_d / (_four * kp.pi * kp.alpha * kp.sqrt_pi * kp.sqrt_alpha)
        * metal::exp(-(norm * norm) / (_four * ttau * kp.alpha));

    *value2 = erf_value / four_pi_alpha_norm;

    *value1 *= kp.alpha_2;
    *value2 *= dot * kp.alpha;
}

inline void d_basis_tri_p1_evaluate_curl_00(
    long i_elem, device const float * n, bool swap, thread float * curls,
    device const long * d_element_nodes,
    device const float * d_node_coords)
{
    device const long * elem_nodes = d_element_nodes + 3 * i_elem;
    device const float * x1rot = d_node_coords + 3 * elem_nodes[0];
    device const float * x2rot = d_node_coords + 3 * elem_nodes[1];
    device const float * x3rot = d_node_coords + 3 * elem_nodes[2];

    float a11 = x2rot[0] - x1rot[0];
    float a12 = x2rot[1] - x1rot[1];
    float a13 = x2rot[2] - x1rot[2];
    float a21 = x3rot[0] - x1rot[0];
    float a22 = x3rot[1] - x1rot[1];
    float a23 = x3rot[2] - x1rot[2];

    float det = n[0] * (a12 * a23 - a13 * a22)
              + n[1] * (a13 * a21 - a11 * a23)
              + n[2] * (a11 * a22 - a21 * a12);

    float g21 =  n[2] * a22 - n[1] * a23;
    float g22 = -n[2] * a21 + n[0] * a23;
    float g23 =  n[1] * a21 - n[0] * a22;

    curls[3] = (n[1] * g23 - n[2] * g22) / det;
    curls[4] = (n[2] * g21 - n[0] * g23) / det;
    curls[5] = (n[0] * g22 - n[1] * g21) / det;

    float g31 = -n[2] * a12 + n[1] * a13;
    float g32 =  n[2] * a11 - n[0] * a13;
    float g33 = -n[1] * a11 + n[0] * a12;

    curls[6] = (n[1] * g33 - n[2] * g32) / det;
    curls[7] = (n[2] * g31 - n[0] * g33) / det;
    curls[8] = (n[0] * g32 - n[1] * g31) / det;

    curls[0] = (-n[1] * (g23 + g33) + n[2] * (g22 + g32)) / det;
    curls[1] = (-n[2] * (g21 + g31) + n[0] * (g23 + g33)) / det;
    curls[2] = (-n[0] * (g22 + g32) + n[1] * (g21 + g31)) / det;
}

template<int quadr_order>
inline void d_get_local_contributions_treg_sreg_sl_p0_p0(
    thread float * values_out,
    long delta, long i_test, long i_trial,
    threadgroup const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_tst,
    thread const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_trl,
    constant mesh_raw_metadata_float & mesh_metadata,
    device const float * d_element_areas,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref,
    constant heat_kernel_parameters_float & kp)
{
    float timestep = mesh_metadata.timestep;
    float ttau = timestep * (float)delta;
    float sqrt_ttau = metal::sqrt(ttau);

    const float test_area = d_element_areas[i_test];
    const float trial_area = d_element_areas[i_trial];

    constant const float * w = quadr_ref._w;

    float value = 0.0f;
    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i_quad = 0; i_quad < quadr_size; ++i_quad) {
        value += d_sl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
            quadr_nodes_tst.xs[i_quad] - quadr_nodes_trl.xs[i_quad],
            quadr_nodes_tst.ys[i_quad] - quadr_nodes_trl.ys[i_quad],
            quadr_nodes_tst.zs[i_quad] - quadr_nodes_trl.zs[i_quad],
            ttau, sqrt_ttau, kp) * w[i_quad];
    }

    float multiplier = test_area * trial_area;
    *values_out = value * multiplier;
}

template<int quadr_order>
inline void d_get_local_contributions_treg_sreg_dl_p0_p1(
    thread float * values_out,
    long delta, long i_test, long i_trial,
    threadgroup const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_tst,
    thread const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_trl,
    constant mesh_raw_metadata_float & mesh_metadata,
    device const float * d_element_areas,
    device const float * d_element_normals,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref,
    constant heat_kernel_parameters_float & kp)
{
    float timestep = mesh_metadata.timestep;
    float ttau = timestep * (float)delta;
    float sqrt_ttau = metal::sqrt(ttau);

    device const float * ny = d_element_normals + 3 * i_trial;

    const float test_area = d_element_areas[i_test];
    const float trial_area = d_element_areas[i_trial];

    constant const float * w = quadr_ref._w;
    constant const float * y1_ref = quadr_ref._y1_ref;
    constant const float * y2_ref = quadr_ref._y2_ref;

    float kernel_val;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;

    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i_quad = 0; i_quad < quadr_size; ++i_quad) {
        kernel_val = d_dl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
            quadr_nodes_tst.xs[i_quad] - quadr_nodes_trl.xs[i_quad],
            quadr_nodes_tst.ys[i_quad] - quadr_nodes_trl.ys[i_quad],
            quadr_nodes_tst.zs[i_quad] - quadr_nodes_trl.zs[i_quad],
            ny, ttau, sqrt_ttau, kp) * w[i_quad];
        value1 += kernel_val * (1.0f - y1_ref[i_quad] - y2_ref[i_quad]);
        value2 += kernel_val * y1_ref[i_quad];
        value3 += kernel_val * y2_ref[i_quad];
    }

    float multiplier = test_area * trial_area;
    values_out[0] = value1 * multiplier;
    values_out[1] = value2 * multiplier;
    values_out[2] = value3 * multiplier;
}

template<int quadr_order>
inline void d_get_local_contributions_treg_sreg_adl_p1_p0(
    thread float * values_out,
    long delta, long i_test, long i_trial,
    threadgroup const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_tst,
    thread const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_trl,
    constant mesh_raw_metadata_float & mesh_metadata,
    device const float * d_element_areas,
    device const float * d_element_normals,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref,
    constant heat_kernel_parameters_float & kp)
{
    float timestep = mesh_metadata.timestep;
    float ttau = timestep * (float)delta;
    float sqrt_ttau = metal::sqrt(ttau);

    device const float * nx = d_element_normals + 3 * i_test;

    const float test_area = d_element_areas[i_test];
    const float trial_area = d_element_areas[i_trial];

    constant const float * w = quadr_ref._w;
    constant const float * x1_ref = quadr_ref._x1_ref;
    constant const float * x2_ref = quadr_ref._x2_ref;

    float kernel_val;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;

    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i_quad = 0; i_quad < quadr_size; ++i_quad) {
        kernel_val = d_adl_kernel_do_anti_tau_anti_t_regular_in_time_regular_in_space(
            quadr_nodes_tst.xs[i_quad] - quadr_nodes_trl.xs[i_quad],
            quadr_nodes_tst.ys[i_quad] - quadr_nodes_trl.ys[i_quad],
            quadr_nodes_tst.zs[i_quad] - quadr_nodes_trl.zs[i_quad],
            nx, ttau, sqrt_ttau, kp) * w[i_quad];
        value1 += kernel_val * (1.0f - x1_ref[i_quad] - x2_ref[i_quad]);
        value2 += kernel_val * x1_ref[i_quad];
        value3 += kernel_val * x2_ref[i_quad];
    }

    float multiplier = test_area * trial_area;
    values_out[0] = value1 * multiplier;
    values_out[1] = value2 * multiplier;
    values_out[2] = value3 * multiplier;
}

template<int quadr_order>
inline void d_get_local_contributions_treg_sreg_hs_p1_p1(
    thread float * values_out,
    long delta, long i_test, long i_trial,
    threadgroup const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_tst,
    thread const quadrature_nodes_raw_float<quadr_order> & quadr_nodes_trl,
    constant mesh_raw_metadata_float & mesh_metadata,
    device const float * d_element_areas,
    device const long * d_element_nodes,
    device const float * d_node_coords,
    device const float * d_element_normals,
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref,
    constant heat_kernel_parameters_float & kp)
{
    float timestep = mesh_metadata.timestep;
    float ttau = timestep * (float)delta;
    float sqrt_ttau = metal::sqrt(ttau);
    device const float * nx = d_element_normals + 3 * i_test;
    device const float * ny = d_element_normals + 3 * i_trial;

    const float test_area = d_element_areas[i_test];
    const float trial_area = d_element_areas[i_trial];

    constant const float * w = quadr_ref._w;
    constant const float * x1_ref = quadr_ref._x1_ref;
    constant const float * x2_ref = quadr_ref._x2_ref;
    constant const float * y1_ref = quadr_ref._y1_ref;
    constant const float * y2_ref = quadr_ref._y2_ref;

    float test_curls[9];
    float trial_curls[9];
    float phi1x, phi1y;
    float kern1, kern2;
    long test_curl_offset, trial_curl_offset;
    float curl_dot[9];

    d_basis_tri_p1_evaluate_curl_00(i_test, nx, false, test_curls, d_element_nodes, d_node_coords);
    d_basis_tri_p1_evaluate_curl_00(i_trial, ny, true, trial_curls, d_element_nodes, d_node_coords);

    for (long i_loc_test = 0; i_loc_test < 3; ++i_loc_test) {
        test_curl_offset = 3 * i_loc_test;
        for (long i_loc_trial = 0; i_loc_trial < 3; ++i_loc_trial) {
            trial_curl_offset = 3 * i_loc_trial;
            curl_dot[i_loc_trial * 3 + i_loc_test]
                = test_curls[test_curl_offset]     * trial_curls[trial_curl_offset]
                + test_curls[test_curl_offset + 1] * trial_curls[trial_curl_offset + 1]
                + test_curls[test_curl_offset + 2] * trial_curls[trial_curl_offset + 2];
        }
    }

    for (long j = 0; j < 9; j++) values_out[j] = 0.0f;
    thread float &value11 = values_out[0]; thread float &value12 = values_out[1]; thread float &value13 = values_out[2];
    thread float &value21 = values_out[3]; thread float &value22 = values_out[4]; thread float &value23 = values_out[5];
    thread float &value31 = values_out[6]; thread float &value32 = values_out[7]; thread float &value33 = values_out[8];

    constexpr int quadr_size = qo2qs(quadr_order);
    for (int i_quad = 0; i_quad < quadr_size; ++i_quad) {
        d_hs_kernel_do_anti_tau_anti_t_and_anti_t_regular_in_time_regular_in_space(
            quadr_nodes_tst.xs[i_quad] - quadr_nodes_trl.xs[i_quad],
            quadr_nodes_tst.ys[i_quad] - quadr_nodes_trl.ys[i_quad],
            quadr_nodes_tst.zs[i_quad] - quadr_nodes_trl.zs[i_quad],
            nx, ny, ttau, sqrt_ttau, &kern1, &kern2, kp);

        phi1x = 1.0f - x1_ref[i_quad] - x2_ref[i_quad];
        phi1y = 1.0f - y1_ref[i_quad] - y2_ref[i_quad];

        value11 += (kern1 * curl_dot[0] + kern2 * phi1x            * phi1y)            * w[i_quad];
        value21 += (kern1 * curl_dot[1] + kern2 * x1_ref[i_quad] * phi1y)            * w[i_quad];
        value31 += (kern1 * curl_dot[2] + kern2 * x2_ref[i_quad] * phi1y)            * w[i_quad];
        value12 += (kern1 * curl_dot[3] + kern2 * phi1x            * y1_ref[i_quad]) * w[i_quad];
        value22 += (kern1 * curl_dot[4] + kern2 * x1_ref[i_quad] * y1_ref[i_quad]) * w[i_quad];
        value32 += (kern1 * curl_dot[5] + kern2 * x2_ref[i_quad] * y1_ref[i_quad]) * w[i_quad];
        value13 += (kern1 * curl_dot[6] + kern2 * phi1x            * y2_ref[i_quad]) * w[i_quad];
        value23 += (kern1 * curl_dot[7] + kern2 * x1_ref[i_quad] * y2_ref[i_quad]) * w[i_quad];
        value33 += (kern1 * curl_dot[8] + kern2 * x2_ref[i_quad] * y2_ref[i_quad]) * w[i_quad];
    }

    float multiplier = test_area * trial_area;
    for (long j = 0; j < 9; j++) values_out[j] *= multiplier;
}

// ----------------------------------------------------------------------------
// Version 2 Compute Kernels
// ----------------------------------------------------------------------------

template<int quadr_order, int tpbx>
kernel void k_apply_ver2_sl_p0_p0(
    device const float * x_perm [[buffer(0)]],
    device float * y [[buffer(1)]],
    constant long & ld_x_perm [[buffer(2)]],
    constant long & ld_y [[buffer(3)]],
    constant float & alpha [[buffer(4)]],
    constant long & i_tst_begin [[buffer(5)]],
    constant mesh_raw_metadata_float & mesh_metadata [[buffer(6)]],
    device const float * d_element_areas [[buffer(7)]],
    device const float * d_node_coords [[buffer(8)]],
    device const long * d_element_nodes [[buffer(9)]],
    device const float * d_element_normals [[buffer(10)]],
    constant heat_kernel_parameters_float & kp [[buffer(11)]],
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim_x [[threads_per_threadgroup]],
    uint block_idx_x [[threadgroup_position_in_grid]])
{
    threadgroup quadrature_nodes_raw_float<quadr_order> shmem_quadr_nodes_tst;
    threadgroup float shmem_matrix_vals[tpbx];

    const long n_blocks = mesh_metadata.n_temporal_elements;
    const long n_elems = mesh_metadata.n_elems;
    const long i_tst = i_tst_begin + block_idx_x;
    const long row = i_tst;

    d_triangles_to_geometry_000_tst_shmem<quadr_order>(i_tst, d_element_nodes, d_node_coords, shmem_quadr_nodes_tst, quadr_ref, tid, block_dim_x);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    quadrature_nodes_raw_float<quadr_order> quadr_nodes_trl;

    float val_prev;
    float val_curr;
    float val_next;

    for (long i = tid; i < n_elems; i += block_dim_x) {
        d_triangles_to_geometry_000_trl<quadr_order>(i, d_element_nodes, d_node_coords, quadr_nodes_trl, quadr_ref);

        val_curr = 0.0f;
        val_next = 0.0f;

        long curr_active_threads =
          (i >= (n_elems / block_dim_x) * block_dim_x) ? (n_elems % block_dim_x) : block_dim_x;

        for (long delta = 0; delta < n_blocks; delta++) {
            {
                long i_trl = i;
                val_prev = val_curr;
                val_curr = val_next;
                d_get_local_contributions_treg_sreg_sl_p0_p0<quadr_order>(&val_next,
                  delta + 1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
                  mesh_metadata, d_element_areas, quadr_ref, kp);
                shmem_matrix_vals[tid] = ((i_tst == i_trl) ? 0.0f : (-val_prev + 2.0f * val_curr - val_next));
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            {
                long max_block = n_blocks - delta;
                for (long block = tid; block < max_block; block += curr_active_threads) {
                    long block_row = delta + block;
                    long block_col = block;
                    float y_val = 0.0f;
                    for (long j = 0; j < curr_active_threads; j++) {
                        long i_trl = (i / block_dim_x) * block_dim_x + j;
                        long col = i_trl;
                        float x_val = x_perm[block_col + ld_x_perm * col];
                        y_val += shmem_matrix_vals[j] * x_val;
                    }
                    y_val *= alpha;
                    y[block_row * ld_y + row] += y_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

template<int quadr_order, int tpbx>
kernel void k_apply_ver2_dl_p0_p1(
    device const float * x_perm [[buffer(0)]],
    device float * y [[buffer(1)]],
    constant long & ld_x_perm [[buffer(2)]],
    constant long & ld_y [[buffer(3)]],
    constant float & alpha [[buffer(4)]],
    constant long & i_tst_begin [[buffer(5)]],
    constant mesh_raw_metadata_float & mesh_metadata [[buffer(6)]],
    device const float * d_element_areas [[buffer(7)]],
    device const float * d_node_coords [[buffer(8)]],
    device const long * d_element_nodes [[buffer(9)]],
    device const float * d_element_normals [[buffer(10)]],
    constant heat_kernel_parameters_float & kp [[buffer(11)]],
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim_x [[threads_per_threadgroup]],
    uint block_idx_x [[threadgroup_position_in_grid]])
{
    threadgroup quadrature_nodes_raw_float<quadr_order> shmem_quadr_nodes_tst;
    threadgroup float shmem_matrix_vals[3][tpbx];

    const long n_blocks = mesh_metadata.n_temporal_elements;
    const long n_elems = mesh_metadata.n_elems;
    const long i_tst = i_tst_begin + block_idx_x;
    const long row = i_tst;

    d_triangles_to_geometry_000_tst_shmem<quadr_order>(i_tst, d_element_nodes, d_node_coords, shmem_quadr_nodes_tst, quadr_ref, tid, block_dim_x);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    quadrature_nodes_raw_float<quadr_order> quadr_nodes_trl;

    float vals_prev[3];
    float vals_curr[3];
    float vals_next[3];

    for (long i = tid; i < n_elems; i += block_dim_x) {
        d_triangles_to_geometry_000_trl<quadr_order>(i, d_element_nodes, d_node_coords, quadr_nodes_trl, quadr_ref);

        vals_curr[0] = 0.0f; vals_curr[1] = 0.0f; vals_curr[2] = 0.0f;
        vals_next[0] = 0.0f; vals_next[1] = 0.0f; vals_next[2] = 0.0f;

        long curr_active_threads =
          (i >= (n_elems / block_dim_x) * block_dim_x) ? (n_elems % block_dim_x) : block_dim_x;

        for (long delta = 0; delta < n_blocks; delta++) {
            {
                long i_trl = i;
                vals_prev[0] = vals_curr[0]; vals_prev[1] = vals_curr[1]; vals_prev[2] = vals_curr[2];
                vals_curr[0] = vals_next[0]; vals_curr[1] = vals_next[1]; vals_curr[2] = vals_next[2];
                d_get_local_contributions_treg_sreg_dl_p0_p1<quadr_order>(vals_next,
                  delta + 1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
                  mesh_metadata, d_element_areas, d_element_normals, quadr_ref, kp);
                shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[0] + 2.0f * vals_curr[0] - vals_next[0]));
                shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[1] + 2.0f * vals_curr[1] - vals_next[1]));
                shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[2] + 2.0f * vals_curr[2] - vals_next[2]));
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            {
                long max_block = n_blocks - delta;
                for (long block = tid; block < max_block; block += curr_active_threads) {
                    long block_row = delta + block;
                    long block_col = block;
                    float y_val = 0.0f;
                    for (long j = 0; j < curr_active_threads; j++) {
                        long i_trl = (i / block_dim_x) * block_dim_x + j;
                        device const long * cols = d_element_nodes + 3 * i_trl;
                        y_val += shmem_matrix_vals[0][j] * x_perm[block_col + ld_x_perm * cols[0]];
                        y_val += shmem_matrix_vals[1][j] * x_perm[block_col + ld_x_perm * cols[1]];
                        y_val += shmem_matrix_vals[2][j] * x_perm[block_col + ld_x_perm * cols[2]];
                    }
                    y_val *= alpha;
                    y[block_row * ld_y + row] += y_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

template<int quadr_order, int tpbx>
kernel void k_apply_ver2_adl_p1_p0(
    device const float * x_perm [[buffer(0)]],
    device float * y [[buffer(1)]],
    constant long & ld_x_perm [[buffer(2)]],
    constant long & ld_y [[buffer(3)]],
    constant float & alpha [[buffer(4)]],
    constant long & i_tst_begin [[buffer(5)]],
    constant mesh_raw_metadata_float & mesh_metadata [[buffer(6)]],
    device const float * d_element_areas [[buffer(7)]],
    device const float * d_node_coords [[buffer(8)]],
    device const long * d_element_nodes [[buffer(9)]],
    device const float * d_element_normals [[buffer(10)]],
    constant heat_kernel_parameters_float & kp [[buffer(11)]],
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim_x [[threads_per_threadgroup]],
    uint block_idx_x [[threadgroup_position_in_grid]])
{
    threadgroup quadrature_nodes_raw_float<quadr_order> shmem_quadr_nodes_tst;
    threadgroup float shmem_matrix_vals[3][tpbx];

    const long n_blocks = mesh_metadata.n_temporal_elements;
    const long n_elems = mesh_metadata.n_elems;
    const long i_tst = i_tst_begin + block_idx_x;
    device const long * rows = d_element_nodes + 3 * i_tst;

    d_triangles_to_geometry_000_tst_shmem<quadr_order>(i_tst, d_element_nodes, d_node_coords, shmem_quadr_nodes_tst, quadr_ref, tid, block_dim_x);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    quadrature_nodes_raw_float<quadr_order> quadr_nodes_trl;

    float vals_prev[3];
    float vals_curr[3];
    float vals_next[3];

    for (long i = tid; i < n_elems; i += block_dim_x) {
        d_triangles_to_geometry_000_trl<quadr_order>(i, d_element_nodes, d_node_coords, quadr_nodes_trl, quadr_ref);

        vals_curr[0] = 0.0f; vals_curr[1] = 0.0f; vals_curr[2] = 0.0f;
        vals_next[0] = 0.0f; vals_next[1] = 0.0f; vals_next[2] = 0.0f;

        long curr_active_threads =
          (i >= (n_elems / block_dim_x) * block_dim_x) ? (n_elems % block_dim_x) : block_dim_x;

        for (long delta = 0; delta < n_blocks; delta++) {
            {
                long i_trl = i;
                vals_prev[0] = vals_curr[0]; vals_prev[1] = vals_curr[1]; vals_prev[2] = vals_curr[2];
                vals_curr[0] = vals_next[0]; vals_curr[1] = vals_next[1]; vals_curr[2] = vals_next[2];
                d_get_local_contributions_treg_sreg_adl_p1_p0<quadr_order>(vals_next,
                  delta + 1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
                  mesh_metadata, d_element_areas, d_element_normals, quadr_ref, kp);
                shmem_matrix_vals[0][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[0] + 2.0f * vals_curr[0] - vals_next[0]));
                shmem_matrix_vals[1][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[1] + 2.0f * vals_curr[1] - vals_next[1]));
                shmem_matrix_vals[2][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[2] + 2.0f * vals_curr[2] - vals_next[2]));
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            {
                long max_block = n_blocks - delta;
                for (long block = tid; block < max_block; block += curr_active_threads) {
                    long block_row = delta + block;
                    long block_col = block;
                    float y_vals[3] = {0.0f, 0.0f, 0.0f};
                    for (long j = 0; j < curr_active_threads; j++) {
                        long i_trl = (i / block_dim_x) * block_dim_x + j;
                        long col = i_trl;
                        float x_val = x_perm[block_col + ld_x_perm * col];
                        y_vals[0] += shmem_matrix_vals[0][j] * x_val;
                        y_vals[1] += shmem_matrix_vals[1][j] * x_val;
                        y_vals[2] += shmem_matrix_vals[2][j] * x_val;
                    }
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[0]], alpha * y_vals[0], memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[1]], alpha * y_vals[1], memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[2]], alpha * y_vals[2], memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

template<int quadr_order, int tpbx>
kernel void k_apply_ver2_hs_p1_p1(
    device const float * x_perm [[buffer(0)]],
    device float * y [[buffer(1)]],
    constant long & ld_x_perm [[buffer(2)]],
    constant long & ld_y [[buffer(3)]],
    constant float & alpha [[buffer(4)]],
    constant long & i_tst_begin [[buffer(5)]],
    constant mesh_raw_metadata_float & mesh_metadata [[buffer(6)]],
    device const float * d_element_areas [[buffer(7)]],
    device const float * d_node_coords [[buffer(8)]],
    device const long * d_element_nodes [[buffer(9)]],
    device const float * d_element_normals [[buffer(10)]],
    constant heat_kernel_parameters_float & kp [[buffer(11)]],
    constant quadrature_reference_raw_float<quadr_order> & quadr_ref [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_dim_x [[threads_per_threadgroup]],
    uint block_idx_x [[threadgroup_position_in_grid]])
{
    threadgroup quadrature_nodes_raw_float<quadr_order> shmem_quadr_nodes_tst;
    threadgroup float shmem_matrix_vals[9][tpbx];

    const long n_blocks = mesh_metadata.n_temporal_elements;
    const long n_elems = mesh_metadata.n_elems;
    const long i_tst = i_tst_begin + block_idx_x;
    device const long * rows = d_element_nodes + 3 * i_tst;

    d_triangles_to_geometry_000_tst_shmem<quadr_order>(i_tst, d_element_nodes, d_node_coords, shmem_quadr_nodes_tst, quadr_ref, tid, block_dim_x);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    quadrature_nodes_raw_float<quadr_order> quadr_nodes_trl;

    float vals_prev[9];
    float vals_curr[9];
    float vals_next[9];

    for (long i = tid; i < n_elems; i += block_dim_x) {
        d_triangles_to_geometry_000_trl<quadr_order>(i, d_element_nodes, d_node_coords, quadr_nodes_trl, quadr_ref);

        for (long j = 0; j < 9; j++) vals_curr[j] = 0.0f;
        for (long j = 0; j < 9; j++) vals_next[j] = 0.0f;

        long curr_active_threads =
          (i >= (n_elems / block_dim_x) * block_dim_x) ? (n_elems % block_dim_x) : block_dim_x;

        for (long delta = 0; delta < n_blocks; delta++) {
            {
                long i_trl = i;
                for (long j = 0; j < 9; j++) vals_prev[j] = vals_curr[j];
                for (long j = 0; j < 9; j++) vals_curr[j] = vals_next[j];
                d_get_local_contributions_treg_sreg_hs_p1_p1<quadr_order>(vals_next,
                  delta + 1, i_tst, i_trl, shmem_quadr_nodes_tst, quadr_nodes_trl,
                  mesh_metadata, d_element_areas, d_element_nodes, d_node_coords, d_element_normals, quadr_ref, kp);
                for (long j = 0; j < 9; j++) {
                    shmem_matrix_vals[j][tid] = ((i_tst == i_trl) ? 0.0f : (-vals_prev[j] + 2.0f * vals_curr[j] - vals_next[j]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            {
                long max_block = n_blocks - delta;
                for (long block = tid; block < max_block; block += curr_active_threads) {
                    long block_row = delta + block;
                    long block_col = block;
                    float y_vals[3] = {0.0f, 0.0f, 0.0f};
                    for (long j = 0; j < curr_active_threads; j++) {
                        long i_trl = (i / block_dim_x) * block_dim_x + j;
                        device const long * cols = d_element_nodes + 3 * i_trl;
                        float x_vals[3];
                        x_vals[0] = x_perm[block_col + ld_x_perm * cols[0]];
                        x_vals[1] = x_perm[block_col + ld_x_perm * cols[1]];
                        x_vals[2] = x_perm[block_col + ld_x_perm * cols[2]];
                        for (int r = 0; r < 3; r++) {
                            for (int c = 0; c < 3; c++) {
                                y_vals[r] += shmem_matrix_vals[3 * r + c][j] * x_vals[c];
                            }
                        }
                    }
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[0]], alpha * y_vals[0], memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[1]], alpha * y_vals[1], memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic<float>*)&y[block_row * ld_y + rows[2]], alpha * y_vals[2], memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel Specializations
// ----------------------------------------------------------------------------

// Single Layer (V)
template [[host_name("k_apply_ver2_sl_p0_p0_qo5")]] kernel void k_apply_ver2_sl_p0_p0<5, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<5> &, uint, uint, uint);

template [[host_name("k_apply_ver2_sl_p0_p0_qo4")]] kernel void k_apply_ver2_sl_p0_p0<4, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<4> &, uint, uint, uint);

template [[host_name("k_apply_ver2_sl_p0_p0_qo2")]] kernel void k_apply_ver2_sl_p0_p0<2, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<2> &, uint, uint, uint);

template [[host_name("k_apply_ver2_sl_p0_p0_qo1")]] kernel void k_apply_ver2_sl_p0_p0<1, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<1> &, uint, uint, uint);

// Double Layer (K)
template [[host_name("k_apply_ver2_dl_p0_p1_qo5")]] kernel void k_apply_ver2_dl_p0_p1<5, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<5> &, uint, uint, uint);

template [[host_name("k_apply_ver2_dl_p0_p1_qo4")]] kernel void k_apply_ver2_dl_p0_p1<4, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<4> &, uint, uint, uint);

template [[host_name("k_apply_ver2_dl_p0_p1_qo2")]] kernel void k_apply_ver2_dl_p0_p1<2, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<2> &, uint, uint, uint);

template [[host_name("k_apply_ver2_dl_p0_p1_qo1")]] kernel void k_apply_ver2_dl_p0_p1<1, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<1> &, uint, uint, uint);

// Adjoint Double Layer (K')
template [[host_name("k_apply_ver2_adl_p1_p0_qo5")]] kernel void k_apply_ver2_adl_p1_p0<5, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<5> &, uint, uint, uint);

template [[host_name("k_apply_ver2_adl_p1_p0_qo4")]] kernel void k_apply_ver2_adl_p1_p0<4, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<4> &, uint, uint, uint);

template [[host_name("k_apply_ver2_adl_p1_p0_qo2")]] kernel void k_apply_ver2_adl_p1_p0<2, 128>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<2> &, uint, uint, uint);

template [[host_name("k_apply_ver2_adl_p1_p0_qo1")]] kernel void k_apply_ver2_adl_p1_p0<1, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<1> &, uint, uint, uint);

// Hypersingular (D)
template [[host_name("k_apply_ver2_hs_p1_p1_qo5")]] kernel void k_apply_ver2_hs_p1_p1<5, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<5> &, uint, uint, uint);

template [[host_name("k_apply_ver2_hs_p1_p1_qo4")]] kernel void k_apply_ver2_hs_p1_p1<4, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<4> &, uint, uint, uint);

template [[host_name("k_apply_ver2_hs_p1_p1_qo2")]] kernel void k_apply_ver2_hs_p1_p1<2, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<2> &, uint, uint, uint);

template [[host_name("k_apply_ver2_hs_p1_p1_qo1")]] kernel void k_apply_ver2_hs_p1_p1<1, 64>(
    device const float *, device float *, constant long &, constant long &, constant float &,
    constant long &, constant mesh_raw_metadata_float &, device const float *, device const float *,
    device const long *, device const float *, constant heat_kernel_parameters_float &,
    constant quadrature_reference_raw_float<1> &, uint, uint, uint);
