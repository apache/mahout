/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

// include necessary system headers
#include <iostream>

#include "viennacl.hpp"
#include "viennacl_private.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"


// IxAMAX

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostiSamax(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                             ViennaCLInt *index,
                                                             float *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostiDamax(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                             ViennaCLInt *index,
                                                             double *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}



// xASUM

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSasum(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDasum(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}



// xAXPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSaxpy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, int incx,
                                                            float *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDaxpy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, int incx,
                                                            double *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostScopy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, int incx,
                                                            float *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 = v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDcopy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, int incx,
                                                            double *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 = v1;
  return ViennaCLSuccess;
}

// xAXPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSdot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           float *alpha,
                                                           float *x, ViennaCLInt offx, int incx,
                                                           float *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDdot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           double *alpha,
                                                           double *x, ViennaCLInt offx, int incx,
                                                           double *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

// xNRM2

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSnrm2(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDnrm2(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}


// xROT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSrot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           float *x, ViennaCLInt offx, int incx,
                                                           float *y, ViennaCLInt offy, int incy,
                                                           float c, float s)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDrot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           double *x, ViennaCLInt offx, int incx,
                                                           double *y, ViennaCLInt offy, int incy,
                                                           double c, double s)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSscal(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  v1 *= alpha;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDscal(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, int incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSswap(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, int incx,
                                                            float *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<float> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDswap(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, int incx,
                                                            double *y, ViennaCLInt offy, int incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, viennacl::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  viennacl::vector_base<double> v2(y, viennacl::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
