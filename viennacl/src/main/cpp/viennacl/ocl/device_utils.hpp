#ifndef VIENNACL_OCL_DEVICE_UTILS_HPP_
#define VIENNACL_OCL_DEVICE_UTILS_HPP_

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

/** @file viennacl/ocl/device_utils.hpp
    @brief Various utility implementations for dispatching with respect to the different devices available on the market.
*/

#define VIENNACL_OCL_MAX_DEVICE_NUM  8

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#include <stddef.h>
#include <map>
#include <string>

#include "viennacl/forwards.h"

namespace viennacl
{
namespace ocl
{

enum vendor_id
{
  beignet_id = 358,
  intel_id = 32902,
  nvidia_id = 4318,
  amd_id = 4098,
  unknown_id = 0
};

//Architecture Family
enum device_architecture_family
{
  //NVidia
  tesla = 0,
  fermi,
  kepler,
  maxwell,

  //AMD
  evergreen,
  northern_islands,
  southern_islands,
  volcanic_islands,

  unknown
};

inline device_architecture_family get_architecture_family(cl_uint vendor_id, std::string const & name)
{
  /*-NVidia-*/
  if (vendor_id==nvidia_id)
  {
    //GeForce
    vcl_size_t found=0;
    if ((found= name.find("GeForce",0)) != std::string::npos)
    {
      if ((found = name.find_first_of("123456789", found)) != std::string::npos)
      {
        switch (name[found])
        {
        case '2' : return tesla;
        case '3' : return tesla;

        case '4' : return fermi;
        case '5' : return fermi;

        case '6' : return kepler;
        case '7' : if (name[found+1] == '5')
                     return maxwell; // GeForce 750 (Ti) are Maxwell-based on desktop GPUs.
                   return kepler;

        case '8' : if (name[found+3] == '0') // Geforce 8600 and friends
                     return tesla;
                   return kepler;  // high-end mobile GPUs tend to be Kepler, low-end GPUs are Maxwell. What a mess. Better be conservative here and pick Kepler.

        case '9' : if (name[found+3] == '0') // Geforce 9600 and friends
                     return tesla;
                   return maxwell;  // GeForce 980, etc.

        default: return unknown;
        }
      }
      else
        return unknown;
    }

    //Tesla
    else if ((found = name.find("Tesla",0)) != std::string::npos)
    {
      if ((found = name.find_first_of("CMK", found)) != std::string::npos)
      {
        switch (name[found])
        {
        case 'C' : return fermi;
        case 'M' : return fermi;
        case 'K' : return kepler;

        default : return unknown;
        }
      }
      else
        return unknown;
    }

    else
      return unknown;
  }

  /*-AMD-*/
  else if (vendor_id==amd_id)
  {

#define VIENNACL_DEVICE_MAP(device,arch)if (name.find(device,0)!=std::string::npos) return arch;

    //Evergreen
    VIENNACL_DEVICE_MAP("Cedar",evergreen);
    VIENNACL_DEVICE_MAP("Redwood",evergreen);
    VIENNACL_DEVICE_MAP("Juniper",evergreen);
    VIENNACL_DEVICE_MAP("Cypress",evergreen);
    VIENNACL_DEVICE_MAP("Hemlock",evergreen);

    //NorthernIslands
    VIENNACL_DEVICE_MAP("Caicos",northern_islands);
    VIENNACL_DEVICE_MAP("Turks",northern_islands);
    VIENNACL_DEVICE_MAP("Barts",northern_islands);
    VIENNACL_DEVICE_MAP("Cayman",northern_islands);
    VIENNACL_DEVICE_MAP("Antilles",northern_islands);

    //SouthernIslands
    VIENNACL_DEVICE_MAP("Cape",southern_islands);
    VIENNACL_DEVICE_MAP("Bonaire",southern_islands);
    VIENNACL_DEVICE_MAP("Pitcairn",southern_islands);
    VIENNACL_DEVICE_MAP("Curacao",southern_islands);
    VIENNACL_DEVICE_MAP("Tahiti",southern_islands);
    VIENNACL_DEVICE_MAP("Malta",southern_islands);
    VIENNACL_DEVICE_MAP("Trinidad",southern_islands);
    VIENNACL_DEVICE_MAP("Tobago",southern_islands);
    VIENNACL_DEVICE_MAP("Oland",southern_islands);

    //VolcanicIslands
    VIENNACL_DEVICE_MAP("Hawaii",volcanic_islands);
    VIENNACL_DEVICE_MAP("Vesuvius",volcanic_islands);
    VIENNACL_DEVICE_MAP("Tonga",volcanic_islands);
    VIENNACL_DEVICE_MAP("Antigua",volcanic_islands);
    VIENNACL_DEVICE_MAP("Grenada",volcanic_islands);
    VIENNACL_DEVICE_MAP("Fiji",volcanic_islands);

    //APUs (map to closest hardware architecture)
    VIENNACL_DEVICE_MAP("Scrapper",northern_islands);
    VIENNACL_DEVICE_MAP("Devastator",northern_islands);

#undef VIENNACL_DEVICE_MAP

    return unknown;

  }

  /*-Other-*/
  else
    return unknown;

}

}
} //namespace viennacl

#endif
