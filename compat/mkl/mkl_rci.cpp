#include "mkl_rci.h"

#include <cmath>
#include <cstring>
#include <vector>

void dcg_init( const long * n, double * /*x*/, const double * /*b*/,
  long * rci_request, long * ipar, double * dpar, double * /*tmp*/ ) {
  *rci_request = 0;
  std::memset( ipar, 0, 128 * sizeof( long ) );
  std::memset( dpar, 0, 128 * sizeof( double ) );

  ipar[ 0 ] = *n;     // problem size
  ipar[ 1 ] = 6;      // screen output
  ipar[ 2 ] = 0;      // current iteration
  ipar[ 3 ] = 0;      // internal state
  ipar[ 4 ] = 1000;   // max iterations
  ipar[ 7 ] = 1;      // do stopping test
  ipar[ 8 ] = 1;      // do residual test
  ipar[ 9 ] = 0;      // no user stopping test
  ipar[ 10 ] = 0;     // non-preconditioned by default

  dpar[ 0 ] = 1e-6;   // rel tol
  dpar[ 2 ] = 0.0;    // initial residual norm
  dpar[ 4 ] = 0.0;    // current residual norm
}

void dcg_check( const long * /*n*/, const double * /*x*/, const double * /*b*/,
  long * rci_request, long * /*ipar*/, double * /*dpar*/, double * /*tmp*/ ) {
  *rci_request = 0;
}

void dcg( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp ) {
  long dim = *n;
  long state = ipar[ 3 ];

  double * p = tmp;                  // size dim (tmp[0 .. dim-1])
  double * Aw = tmp + dim;           // size dim (tmp[dim .. 2*dim-1])
  double * r = tmp + 2 * dim;        // size dim (tmp[2*dim .. 3*dim-1])

  if ( state == 0 ) {
    // Stage 0: Need A*x0 to compute initial residual r = b - A*x0
    std::memcpy( p, x, dim * sizeof( double ) );
    *rci_request = 1;  // apply A
    ipar[ 3 ] = 1;
    return;
  } else if ( state == 1 ) {
    // Stage 1: Aw holds A*x0. Compute r = b - Aw.
    double norm_sq = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      r[ i ] = b[ i ] - Aw[ i ];
      norm_sq += r[ i ] * r[ i ];
    }
    double norm = std::sqrt( norm_sq );
    dpar[ 2 ] = norm;  // initial residual norm
    dpar[ 4 ] = norm;  // current residual norm

    if ( norm < 1e-30 || dpar[ 4 ] <= dpar[ 0 ] * dpar[ 2 ] ) {
      *rci_request = 0;
      ipar[ 3 ] = 99;
      return;
    }

    if ( ipar[ 10 ] == 0 ) {
      // Non-preconditioned: p = r, rho = r.r
      std::memcpy( p, r, dim * sizeof( double ) );
      dpar[ 5 ] = norm_sq;  // rho
      *rci_request = 1;     // apply A(p)
      ipar[ 3 ] = 3;
      return;
    } else {
      // Preconditioned: request M^{-1}*r
      std::memcpy( p, r, dim * sizeof( double ) );
      *rci_request = 3;     // apply preconditioner
      ipar[ 3 ] = 2;
      return;
    }
  } else if ( state == 2 ) {
    // Stage 2: Aw holds z = M^{-1}*r.
    // p = z, rho = r.z
    double rho = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      p[ i ] = Aw[ i ];
      rho += r[ i ] * p[ i ];
    }
    dpar[ 5 ] = rho;
    *rci_request = 1;  // apply A(p)
    ipar[ 3 ] = 3;
    return;
  } else if ( state == 3 ) {
    // Stage 3: Aw holds A*p.
    double p_Ap = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      p_Ap += p[ i ] * Aw[ i ];
    }
    if ( std::abs( p_Ap ) < 1e-35 ) {
      *rci_request = -1;  // breakdown
      return;
    }
    double alpha = dpar[ 5 ] / p_Ap;

    double norm_sq = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      x[ i ] += alpha * p[ i ];
      r[ i ] -= alpha * Aw[ i ];
      norm_sq += r[ i ] * r[ i ];
    }
    double norm = std::sqrt( norm_sq );
    dpar[ 4 ] = norm;
    ipar[ 2 ]++;  // iter++

    if ( dpar[ 4 ] <= dpar[ 0 ] * dpar[ 2 ] || ipar[ 2 ] >= ipar[ 4 ] ) {
      *rci_request = 0;
      ipar[ 3 ] = 99;
      return;
    }

    if ( ipar[ 10 ] == 0 ) {
      // Non-preconditioned
      double rho_new = norm_sq;
      double beta = rho_new / dpar[ 5 ];
      dpar[ 5 ] = rho_new;
      for ( long i = 0; i < dim; ++i ) {
        p[ i ] = r[ i ] + beta * p[ i ];
      }
      *rci_request = 1;  // apply A(p)
      ipar[ 3 ] = 3;
      return;
    } else {
      // Preconditioned
      std::memcpy( p, r, dim * sizeof( double ) );
      *rci_request = 3;  // apply preconditioner
      ipar[ 3 ] = 4;
      return;
    }
  } else if ( state == 4 ) {
    // Stage 4: Aw holds z = M^{-1}*r
    double rho_new = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      rho_new += r[ i ] * Aw[ i ];
    }
    double beta = rho_new / dpar[ 5 ];
    dpar[ 5 ] = rho_new;
    for ( long i = 0; i < dim; ++i ) {
      p[ i ] = Aw[ i ] + beta * p[ i ];
    }
    *rci_request = 1;  // apply A(p)
    ipar[ 3 ] = 3;
    return;
  }

  *rci_request = 0;
}

void dcg_get( const long * /*n*/, const double * /*x*/, const double * /*b*/,
  long * rci_request, const long * ipar, const double * /*dpar*/,
  double * /*tmp*/, long * itercount ) {
  *itercount = ipar[ 2 ];
  *rci_request = 0;
}

// ---------------------------------------------------------------------------
// DFGMRES implementation
// ---------------------------------------------------------------------------

struct fgmres_ctx {
  long dim;
  long m;       // restart limit
  long j;       // current step in [0, m-1]
  long total_iter;
  double r0_norm;
  std::vector< double > s;         // rhs of least squares problem (m+1)
  std::vector< double > cs;        // Givens cosines (m)
  std::vector< double > sn;        // Givens sines (m)
  std::vector< double > H;         // Hessenberg matrix (m+1) x m
  std::vector< double > x0;        // initial solution at start of restart cycle
};

static fgmres_ctx g_fgmres_ctx;

void dfgmres_init( const long * n, double * /*x*/, const double * /*b*/,
  long * rci_request, long * ipar, double * dpar, double * /*tmp*/ ) {
  *rci_request = 0;
  std::memset( ipar, 0, 128 * sizeof( long ) );
  std::memset( dpar, 0, 128 * sizeof( double ) );

  ipar[ 0 ] = *n;
  ipar[ 1 ] = 6;
  ipar[ 2 ] = 0;
  ipar[ 3 ] = 0;
  ipar[ 4 ] = 1000;
  ipar[ 7 ] = 1;
  ipar[ 8 ] = 1;
  ipar[ 9 ] = 0;
  ipar[ 10 ] = 0;
  ipar[ 11 ] = 1;
  ipar[ 14 ] = 30;  // default restart

  dpar[ 0 ] = 1e-6;
}

void dfgmres_check( const long * /*n*/, const double * /*x*/, const double * /*b*/,
  long * rci_request, long * /*ipar*/, double * /*dpar*/, double * /*tmp*/ ) {
  *rci_request = 0;
}

void dfgmres( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp ) {
  long dim = *n;
  long m = ipar[ 14 ];
  if ( m <= 0 ) m = 30;

  long state = ipar[ 3 ];

  // Layout in tmp according to MKL dfgmres standard:
  // V vectors: 0 .. (m) * dim
  // Z vectors: (m+1)*dim .. (2m+1)*dim
  // w (residual/Ax): (2m+1)*dim .. (2m+2)*dim
  long v_offset = 0;
  long z_offset = ( m + 1 ) * dim;
  long w_offset = ( 2 * m + 1 ) * dim;

  if ( state == 0 ) {
    g_fgmres_ctx.dim = dim;
    g_fgmres_ctx.m = m;
    g_fgmres_ctx.j = 0;
    g_fgmres_ctx.total_iter = 0;
    g_fgmres_ctx.s.assign( m + 1, 0.0 );
    g_fgmres_ctx.cs.assign( m, 0.0 );
    g_fgmres_ctx.sn.assign( m, 0.0 );
    g_fgmres_ctx.H.assign( ( m + 1 ) * m, 0.0 );
    g_fgmres_ctx.x0.assign( x, x + dim );

    // Step 0: compute A*x0
    std::memcpy( tmp + w_offset, x, dim * sizeof( double ) );
    ipar[ 21 ] = w_offset + 1;  // input to A
    ipar[ 22 ] = w_offset + 1;  // output from A
    *rci_request = 1;
    ipar[ 3 ] = 1;
    return;
  } else if ( state == 1 ) {
    // tmp[w_offset] has A*x0. Compute r0 = b - A*x0.
    double * r0 = tmp + v_offset;  // store v0 in V[:, 0]
    double norm_sq = 0.0;
    for ( long i = 0; i < dim; ++i ) {
      r0[ i ] = b[ i ] - tmp[ w_offset + i ];
      norm_sq += r0[ i ] * r0[ i ];
    }
    double beta_norm = std::sqrt( norm_sq );
    if ( g_fgmres_ctx.total_iter == 0 ) {
      g_fgmres_ctx.r0_norm = beta_norm;
      dpar[ 2 ] = beta_norm;
    }
    dpar[ 4 ] = beta_norm;

    if ( beta_norm < 1e-30 || dpar[ 4 ] <= dpar[ 0 ] * dpar[ 2 ] ) {
      *rci_request = 0;
      ipar[ 3 ] = 99;
      return;
    }

    // Normalize v0
    for ( long i = 0; i < dim; ++i ) {
      r0[ i ] /= beta_norm;
    }
    g_fgmres_ctx.s[ 0 ] = beta_norm;
    for ( long i = 1; i <= m; ++i ) g_fgmres_ctx.s[ i ] = 0.0;
    g_fgmres_ctx.j = 0;

    // Next: compute z_j = M^{-1} v_j (or z_j = v_j if no preconditioner)
    if ( ipar[ 10 ] == 0 ) {
      // Non-preconditioned: z_j = v_j
      std::memcpy( tmp + z_offset, r0, dim * sizeof( double ) );
      ipar[ 21 ] = z_offset + 1;
      ipar[ 22 ] = w_offset + 1;
      *rci_request = 1;  // compute A*z_j
      ipar[ 3 ] = 3;
      return;
    } else {
      // Preconditioned
      ipar[ 21 ] = v_offset + 1;
      ipar[ 22 ] = z_offset + 1;
      *rci_request = 3;  // compute M^{-1}*v_j
      ipar[ 3 ] = 2;
      return;
    }
  } else if ( state == 2 ) {
    // tmp[z_offset + j*dim] has z_j. Now compute A*z_j.
    long j = g_fgmres_ctx.j;
    ipar[ 21 ] = z_offset + j * dim + 1;
    ipar[ 22 ] = w_offset + 1;
    *rci_request = 1;  // compute A*z_j
    ipar[ 3 ] = 3;
    return;
  } else if ( state == 3 ) {
    // tmp[w_offset] has w = A*z_j.
    // Modified Gram-Schmidt orthogonalization against V[:, 0..j]
    long j = g_fgmres_ctx.j;
    double * w = tmp + w_offset;
    for ( long i = 0; i <= j; ++i ) {
      double * vi = tmp + v_offset + i * dim;
      double hij = 0.0;
      for ( long k = 0; k < dim; ++k ) hij += w[ k ] * vi[ k ];
      g_fgmres_ctx.H[ i * m + j ] = hij;
      for ( long k = 0; k < dim; ++k ) w[ k ] -= hij * vi[ k ];
    }
    double w_norm = 0.0;
    for ( long k = 0; k < dim; ++k ) w_norm += w[ k ] * w[ k ];
    w_norm = std::sqrt( w_norm );
    g_fgmres_ctx.H[ ( j + 1 ) * m + j ] = w_norm;

    // Apply previous Givens rotations to column j
    for ( long i = 0; i < j; ++i ) {
      double h0 = g_fgmres_ctx.H[ i * m + j ];
      double h1 = g_fgmres_ctx.H[ ( i + 1 ) * m + j ];
      double c = g_fgmres_ctx.cs[ i ];
      double s = g_fgmres_ctx.sn[ i ];
      g_fgmres_ctx.H[ i * m + j ] = c * h0 - s * h1;
      g_fgmres_ctx.H[ ( i + 1 ) * m + j ] = s * h0 + c * h1;
    }

    // Compute new Givens rotation for row j, j+1
    double a = g_fgmres_ctx.H[ j * m + j ];
    double b_elem = g_fgmres_ctx.H[ ( j + 1 ) * m + j ];
    double r = std::sqrt( a * a + b_elem * b_elem );
    double c = 1.0, s_rot = 0.0;
    if ( r > 1e-35 ) {
      c = a / r;
      s_rot = -b_elem / r;
    }
    g_fgmres_ctx.cs[ j ] = c;
    g_fgmres_ctx.sn[ j ] = s_rot;

    g_fgmres_ctx.H[ j * m + j ] = c * a - s_rot * b_elem;
    g_fgmres_ctx.H[ ( j + 1 ) * m + j ] = 0.0;

    double s0 = g_fgmres_ctx.s[ j ];
    g_fgmres_ctx.s[ j ] = c * s0;
    g_fgmres_ctx.s[ j + 1 ] = s_rot * s0;

    double res_norm = std::abs( g_fgmres_ctx.s[ j + 1 ] );
    dpar[ 4 ] = res_norm;
    g_fgmres_ctx.total_iter++;
    ipar[ 2 ] = g_fgmres_ctx.total_iter;

    // Store v_{j+1} if w_norm > 0
    if ( j + 1 <= m && w_norm > 1e-35 ) {
      double * vj1 = tmp + v_offset + ( j + 1 ) * dim;
      for ( long k = 0; k < dim; ++k ) vj1[ k ] = w[ k ] / w_norm;
    }

    bool converged = ( res_norm <= dpar[ 0 ] * dpar[ 2 ] );
    bool max_iter = ( g_fgmres_ctx.total_iter >= ipar[ 4 ] );

    if ( converged || max_iter || ( j + 1 >= m ) ) {
      // Solve triangular system H * y = s
      long num_steps = j + 1;
      std::vector< double > y_ls( num_steps, 0.0 );
      for ( long i = num_steps - 1; i >= 0; --i ) {
        double sum = g_fgmres_ctx.s[ i ];
        for ( long k = i + 1; k < num_steps; ++k ) {
          sum -= g_fgmres_ctx.H[ i * m + k ] * y_ls[ k ];
        }
        y_ls[ i ] = ( std::abs( g_fgmres_ctx.H[ i * m + i ] ) > 1e-35 )
          ? ( sum / g_fgmres_ctx.H[ i * m + i ] ) : 0.0;
      }
      // Update x = x0 + Z * y_ls
      for ( long k = 0; k < dim; ++k ) x[ k ] = g_fgmres_ctx.x0[ k ];
      for ( long i = 0; i < num_steps; ++i ) {
        double * zi = tmp + z_offset + i * dim;
        double yi = y_ls[ i ];
        for ( long k = 0; k < dim; ++k ) x[ k ] += yi * zi[ k ];
      }

      if ( converged || max_iter ) {
        *rci_request = 0;
        ipar[ 3 ] = 99;
        return;
      }

      // Restart cycle
      g_fgmres_ctx.x0.assign( x, x + dim );
      std::memcpy( tmp + w_offset, x, dim * sizeof( double ) );
      ipar[ 21 ] = w_offset + 1;
      ipar[ 22 ] = w_offset + 1;
      *rci_request = 1;
      ipar[ 3 ] = 1;
      return;
    }

    // Next step in current Krylov cycle
    g_fgmres_ctx.j++;
    j = g_fgmres_ctx.j;
    if ( ipar[ 10 ] == 0 ) {
      // Non-preconditioned: z_j = v_j
      std::memcpy( tmp + z_offset + j * dim, tmp + v_offset + j * dim,
        dim * sizeof( double ) );
      ipar[ 21 ] = z_offset + j * dim + 1;
      ipar[ 22 ] = w_offset + 1;
      *rci_request = 1;
      ipar[ 3 ] = 3;
      return;
    } else {
      ipar[ 21 ] = v_offset + j * dim + 1;
      ipar[ 22 ] = z_offset + j * dim + 1;
      *rci_request = 3;
      ipar[ 3 ] = 2;
      return;
    }
  }

  *rci_request = 0;
}

void dfgmres_get( const long * /*n*/, const double * /*x*/, const double * /*b*/,
  long * rci_request, const long * ipar, const double * /*dpar*/,
  double * /*tmp*/, long * itercount ) {
  *itercount = ipar[ 2 ];
  *rci_request = 0;
}
