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



static ViennaCLStatus init_cuda_vector(viennacl::backend::mem_handle & h, ViennaCLVector x)
{
#ifdef VIENNACL_WITH_CUDA
  h.switch_active_handle_id(viennacl::CUDA_MEMORY);
  h.cuda_handle().reset(x->cuda_mem);
  h.cuda_handle().inc();
  if (x->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * x->size * sizeof(float)); // not necessary, but still set for conciseness
  else if (x->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * x->size * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
#else
  (void)h;
  (void)x;
  return ViennaCLGenericFailure;
#endif
}

static ViennaCLStatus init_opencl_vector(viennacl::backend::mem_handle & h, ViennaCLVector x)
{
#ifdef VIENNACL_WITH_OPENCL
  h.switch_active_handle_id(viennacl::OPENCL_MEMORY);
  h.opencl_handle() = x->opencl_mem;
  h.opencl_handle().inc();
  if (x->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * static_cast<viennacl::vcl_size_t>(x->size) * sizeof(float)); // not necessary, but still set for conciseness
  else if (x->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * static_cast<viennacl::vcl_size_t>(x->size) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
#else
  (void)h;
  (void)x;
  return ViennaCLGenericFailure;
#endif
}


static ViennaCLStatus init_host_vector(viennacl::backend::mem_handle & h, ViennaCLVector x)
{
  h.switch_active_handle_id(viennacl::MAIN_MEMORY);
  h.ram_handle().reset(x->host_mem);
  h.ram_handle().inc();
  if (x->precision == ViennaCLFloat)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * static_cast<viennacl::vcl_size_t>(x->size) * sizeof(float)); // not necessary, but still set for conciseness
  else if (x->precision == ViennaCLDouble)
    h.raw_size(static_cast<viennacl::vcl_size_t>(x->inc) * static_cast<viennacl::vcl_size_t>(x->size) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return ViennaCLGenericFailure;

  return ViennaCLSuccess;
}


static ViennaCLStatus init_vector(viennacl::backend::mem_handle & h, ViennaCLVector x)
{
  switch (x->backend->backend_type)
  {
    case ViennaCLCUDA:
      return init_cuda_vector(h, x);

    case ViennaCLOpenCL:
      return init_opencl_vector(h, x);

    case ViennaCLHost:
      return init_host_vector(h, x);

    default:
      return ViennaCLGenericFailure;
  }
}



