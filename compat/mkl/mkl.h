#ifndef BESTHEA_COMPAT_MKL_H_
#define BESTHEA_COMPAT_MKL_H_

#include <cblas.h>
#include <lapacke.h>

#include <cstdint>

#ifndef MKL_INT
#define MKL_INT long
#endif

// Wrappers for LAPACK routines with MKL_INT / long signatures
inline void dgeqrf( const long * m, const long * n, double * a,
  const long * lda, double * tau, double * work, const long * lwork,
  long * info ) {
  lapack_int m_ = *m;
  lapack_int n_ = *n;
  lapack_int lda_ = *lda;
  lapack_int lwork_ = *lwork;
  lapack_int info_ = 0;
  LAPACK_dgeqrf( &m_, &n_, a, &lda_, tau, work, &lwork_, &info_ );
  *info = info_;
}

inline void dorgqr( const long * m, const long * n, const long * k,
  double * a, const long * lda, const double * tau, double * work,
  const long * lwork, long * info ) {
  lapack_int m_ = *m;
  lapack_int n_ = *n;
  lapack_int k_ = *k;
  lapack_int lda_ = *lda;
  lapack_int lwork_ = *lwork;
  lapack_int info_ = 0;
  LAPACK_dorgqr( &m_, &n_, &k_, a, &lda_, tau, work, &lwork_, &info_ );
  *info = info_;
}

inline void dgesvd( const char * jobu, const char * jobvt, const long * m,
  const long * n, double * a, const long * lda, double * s, double * u,
  const long * ldu, double * vt, const long * ldvt, double * work,
  const long * lwork, long * info ) {
  lapack_int m_ = *m;
  lapack_int n_ = *n;
  lapack_int lda_ = *lda;
  lapack_int ldu_ = *ldu;
  lapack_int ldvt_ = *ldvt;
  lapack_int lwork_ = *lwork;
  lapack_int info_ = 0;
  LAPACK_dgesvd( jobu, jobvt, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
    work, &lwork_, &info_ );
  *info = info_;
}

#endif  // BESTHEA_COMPAT_MKL_H_
