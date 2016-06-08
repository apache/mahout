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

#include "viennacl.hpp"
#include "viennacl/backend/mem_handle.hpp"



static ViennaCLStatus init_cuda_matrix(viennacl::backend::mem_handle & h, ViennaCLMatrix A)
{
#ifdef VIENNACL_WITH_CUDA
  h.switch_active_handle_id(viennacl::CUDA_MEMORY);
  h.cuda_handle().reset(A->cuda_mem);
  h.cuda_handle().inc();
  if (A->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(float)); // not necessary, but still set for conciseness
  else if (A->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
#else
  (void)h;
  (void)A;
  return ViennaCLGenericFailure;
#endif
}

static ViennaCLStatus init_opencl_matrix(viennacl::backend::mem_handle & h, ViennaCLMatrix A)
{
#ifdef VIENNACL_WITH_OPENCL
  h.switch_active_handle_id(viennacl::OPENCL_MEMORY);
  h.opencl_handle() = A->opencl_mem;
  h.opencl_handle().inc();
  if (A->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(float)); // not necessary, but still set for conciseness
  else if (A->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
#else
  (void)h;
  (void)A;
  return ViennaCLGenericFailure;
#endif
}


static ViennaCLStatus init_host_matrix(viennacl::backend::mem_handle & h, ViennaCLMatrix A)
{
  h.switch_active_handle_id(viennacl::MAIN_MEMORY);
  h.ram_handle().reset(A->host_mem);
  h.ram_handle().inc();
  if (A->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(float)); // not necessary, but still set for conciseness
  else if (A->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(A->internal_size1) * static_cast<viennacl::vcl_size_t>(A->internal_size2) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
}


static ViennaCLStatus init_matrix(viennacl::backend::mem_handle & h, ViennaCLMatrix A)
{
  switch (A->backend->backend_type)
  {
    case ViennaCLCUDA:
      return init_cuda_matrix(h, A);

    case ViennaCLOpenCL:
      return init_opencl_matrix(h, A);

    case ViennaCLHost:
      return init_host_matrix(h, A);

    default:
      return ViennaCLGenericFailure;
  }
}



