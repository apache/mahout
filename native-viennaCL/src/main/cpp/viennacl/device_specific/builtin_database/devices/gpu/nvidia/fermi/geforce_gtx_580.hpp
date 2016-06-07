#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_FERMI_GTX580_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_FERMI_GTX580_HPP_

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#include "viennacl/device_specific/templates/matrix_product_template.hpp"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/builtin_database/common.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{
namespace devices{
namespace gpu{
namespace nvidia{
namespace fermi{
namespace geforce_gtx_580{

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,2,1,128,4,1,4,FETCH_FROM_GLOBAL_CONTIGUOUS,FETCH_FROM_GLOBAL_STRIDED,0,0));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,16,128,32,2,4,2,FETCH_FROM_LOCAL,FETCH_FROM_GLOBAL_STRIDED,32,16));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,4,64,128,4,1,2,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_GLOBAL_STRIDED,0,0));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,128,32,1,1,1,8,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_LOCAL,32,4));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,2,32,32,8,1,2,FETCH_FROM_LOCAL,FETCH_FROM_GLOBAL_STRIDED,32,2));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,16,32,16,4,4,2,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,32,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(2,16,16,16,4,4,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,16,16));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GTX 580", matrix_product_template::parameters_type(1,128,16,2,4,1,8,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_GLOBAL_CONTIGUOUS,0,0));
}


}
}
}
}
}
}
}
}
#endif
