#ifndef BESTHEA_COMPAT_MKL_RCI_H_
#define BESTHEA_COMPAT_MKL_RCI_H_

#ifdef __cplusplus
extern "C" {
#endif

void dcg_init( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dcg_check( const long * n, const double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dcg( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dcg_get( const long * n, const double * x, const double * b,
  long * rci_request, const long * ipar, const double * dpar,
  double * tmp, long * itercount );

void dfgmres_init( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dfgmres_check( const long * n, const double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dfgmres( const long * n, double * x, const double * b,
  long * rci_request, long * ipar, double * dpar, double * tmp );

void dfgmres_get( const long * n, const double * x, const double * b,
  long * rci_request, const long * ipar, const double * dpar,
  double * tmp, long * itercount );

#ifdef __cplusplus
}
#endif

#endif  // BESTHEA_COMPAT_MKL_RCI_H_
