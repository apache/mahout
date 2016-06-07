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


VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendCreate(ViennaCLBackend * backend)
{
  *backend = new ViennaCLBackend_impl();

  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendSetOpenCLContextID(ViennaCLBackend backend, ViennaCLInt context_id)
{
  backend->opencl_backend.context_id = context_id;

  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendDestroy(ViennaCLBackend * backend)
{
  delete *backend;
  *backend = NULL;

  return ViennaCLSuccess;
}

