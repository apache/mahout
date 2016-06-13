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

#ifdef VIENNACL_WITH_OPENCL

// IxAMAX

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                                               ViennaCLInt *index,
                                                               cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                                               ViennaCLInt *index,
                                                               cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *index = static_cast<ViennaCLInt>(viennacl::linalg::index_norm_inf(v1));
  return ViennaCLSuccess;
}




// xASUM

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLBackend backend, ViennaCLInt n,
                                                              float *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLBackend backend, ViennaCLInt n,
                                                              double *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_1(v1);
  return ViennaCLSuccess;
}



// xAXPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                              float alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                              double alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 += alpha * v1;
  return ViennaCLSuccess;
}


// xCOPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v2 = v1;
  return ViennaCLSuccess;
}

// xDOT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLBackend backend, ViennaCLInt n,
                                                             float *alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLBackend backend, ViennaCLInt n,
                                                             double *alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::inner_prod(v1, v2);
  return ViennaCLSuccess;
}


// xNRM2

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                              float *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                              double *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  *alpha = viennacl::linalg::norm_2(v1);
  return ViennaCLSuccess;
}


// xROT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLBackend backend, ViennaCLInt n,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             float c, float s)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLBackend backend, ViennaCLInt n,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             double c, double s)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::linalg::plane_rotation(v1, v2, c, s);
  return ViennaCLSuccess;
}



// xSCAL

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLBackend backend, ViennaCLInt n,
                                                              float alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLBackend backend, ViennaCLInt n,
                                                              double alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  v1 *= alpha;
  return ViennaCLSuccess;
}

// xSWAP

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<float>::size_type           size_type;
  typedef viennacl::vector_base<float>::size_type           difference_type;
  viennacl::vector_base<float> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<float> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy)
{
  typedef viennacl::vector_base<double>::size_type           size_type;
  typedef viennacl::vector_base<double>::size_type           difference_type;
  viennacl::vector_base<double> v1(x, size_type(n), size_type(offx), difference_type(incx), viennacl::ocl::get_context(backend->opencl_backend.context_id));
  viennacl::vector_base<double> v2(y, size_type(n), size_type(offy), difference_type(incy), viennacl::ocl::get_context(backend->opencl_backend.context_id));

  viennacl::swap(v1, v2);
  return ViennaCLSuccess;
}
#endif
