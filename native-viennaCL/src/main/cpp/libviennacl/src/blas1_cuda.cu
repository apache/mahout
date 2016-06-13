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


#ifdef VIENNACL_WITH_CUDA


// IxAMAX

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAiSamax(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                             ViennaCLInt *index,
                                                             float *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAiDamax(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                             ViennaCLInt *index,
                                                             double *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}



// xASUM

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASasum(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADasum(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}


// xAXPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASaxpy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADaxpy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAScopy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADcopy(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  v2 = v1;
  return ViennaCLSuccess;
}

// xDOT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASdot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           float *alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADdot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           double *alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

// xNRM2

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASnrm2(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADnrm2(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}



// xROT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASrot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float c, float s)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADrot(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double c, double s)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASscal(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADscal(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);

  v1 *= alpha;
  return ViennaCLSuccess;
}


// xSWAP

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASswap(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADswap(ViennaCLBackend /*backend*/, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, n, offy, incy);

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
#endif


