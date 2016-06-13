#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_

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

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/barts.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/tesla_c2050.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_470.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell/geforce_gtx_750_ti.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/scrapper.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla/geforce_gtx_260.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands/tahiti.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/devastator.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler/tesla_k20m.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_580.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands/hawaii.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cypress.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cedar.hpp"


#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"


#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"

/** @file viennacl/device_specific/builtin_database/matrix_product.hpp
*
* Initializes the device database with the provided profiles. Updated semi-automatically.
*/

namespace viennacl
{
namespace device_specific
{
namespace builtin_database
{

inline database_type<matrix_product_template::parameters_type> init_matrix_product_N_N()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  devices::gpu::amd::evergreen::cedar::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::evergreen::cypress::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::southern_islands::tahiti::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::southern_islands::tahiti::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::devastator::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::scrapper::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::barts::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());

  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_T_N()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::evergreen::cedar::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::evergreen::cypress::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::southern_islands::tahiti::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::southern_islands::tahiti::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::devastator::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::scrapper::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::amd::northern_islands::barts::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());

  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_N_T()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::evergreen::cedar::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::evergreen::cypress::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::southern_islands::tahiti::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::southern_islands::tahiti::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::northern_islands::devastator::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());


  devices::gpu::amd::northern_islands::scrapper::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::amd::northern_islands::barts::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());

  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_T_T()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::evergreen::cedar::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::evergreen::cypress::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::volcanic_islands::hawaii::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::kepler::tesla_k20m::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::southern_islands::tahiti::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::southern_islands::tahiti::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::northern_islands::devastator::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::northern_islands::scrapper::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::nvidia::fermi::tesla_c2050::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::amd::northern_islands::barts::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());

  return result;
}

static database_type<matrix_product_template::parameters_type> matrix_product_N_N = init_matrix_product_N_N();
static database_type<matrix_product_template::parameters_type> matrix_product_T_N = init_matrix_product_T_N();
static database_type<matrix_product_template::parameters_type> matrix_product_N_T = init_matrix_product_N_T();
static database_type<matrix_product_template::parameters_type> matrix_product_T_T = init_matrix_product_T_T();

template<class NumericT>
matrix_product_template::parameters_type const & matrix_product_params(ocl::device const & device, char A_trans, char B_trans)
{
  assert(A_trans=='N' || A_trans=='T');
  assert(B_trans=='N' || B_trans=='T');
  database_type<matrix_product_template::parameters_type> * db;
  if (A_trans=='N' && B_trans=='N')
    db = &matrix_product_N_N;
  else if (A_trans=='T' && B_trans=='N')
    db = &matrix_product_T_N;
  else if (A_trans=='N' && B_trans=='T')
    db = &matrix_product_N_T;
  else
    db = &matrix_product_T_T;
  return get_parameters<NumericT>(*db, device);
}


}
}
}
#endif
