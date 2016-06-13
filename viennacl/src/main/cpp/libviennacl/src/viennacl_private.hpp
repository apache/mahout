#ifndef VIENNACL_VIENNACL_PRIVATE_HPP
#define VIENNACL_VIENNACL_PRIVATE_HPP


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

#include <stdlib.h>

#ifdef VIENNACL_WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

#include "viennacl.hpp"


/************* Backend Management ******************/

struct ViennaCLCUDABackend_impl
{
    //TODO: Add stream and/or device descriptors here
};

struct ViennaCLOpenCLBackend_impl
{
  ViennaCLInt context_id;
};

struct ViennaCLHostBackend_impl
{
  // Nothing to specify *at the moment*
};


/** @brief Generic backend for CUDA, OpenCL, host-based stuff */
struct ViennaCLBackend_impl
{
  ViennaCLBackendTypes backend_type;

  ViennaCLCUDABackend_impl     cuda_backend;
  ViennaCLOpenCLBackend_impl   opencl_backend;
  ViennaCLHostBackend_impl     host_backend;
};



/******** User Types **********/

struct ViennaCLHostScalar_impl
{
  ViennaCLPrecision  precision;

  union {
    float  value_float;
    double value_double;
  };
};

struct ViennaCLScalar_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  ViennaCLInt   offset;
};

struct ViennaCLVector_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  ViennaCLInt   offset;
  ViennaCLInt   inc;
  ViennaCLInt   size;
};

struct ViennaCLMatrix_impl
{
  ViennaCLBackend    backend;
  ViennaCLPrecision  precision;
  ViennaCLOrder      order;
  ViennaCLTranspose  trans;

  // buffer:
#ifdef VIENNACL_WITH_CUDA
  char * cuda_mem;
#endif
#ifdef VIENNACL_WITH_OPENCL
  cl_mem opencl_mem;
#endif
  char * host_mem;

  ViennaCLInt   size1;
  ViennaCLInt   start1;
  ViennaCLInt   stride1;
  ViennaCLInt   internal_size1;

  ViennaCLInt   size2;
  ViennaCLInt   start2;
  ViennaCLInt   stride2;
  ViennaCLInt   internal_size2;
};


#endif
