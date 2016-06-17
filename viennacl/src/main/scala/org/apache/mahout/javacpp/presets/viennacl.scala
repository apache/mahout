/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.javacpp.presets

import org.bytedeco.javacpp.{Loader, Pointer}
import org.bytedeco.javacpp.annotation.{Namespace, Platform, _}
import org.bytedeco.javacpp.tools.{Info, InfoMap, InfoMapper}

@Platform(includepath=Array("/usr/include/","/usr/include/CL/","/usr/include/viennacl/"),
  include=Array(viennacl.PLATFORM_HEADER_FILE_ONE,viennacl.PLATFORM_HEADER_FILE_TWO,viennacl.PLATFORM_HEADER_FILE_THREE,
    viennacl.PLATFORM_HEADER_FILE_FOUR, viennacl.PLATFORM_HEADER_FILE_FIVE, viennacl.PLATFORM_HEADER_FILE_SIX,
    viennacl.PLATFORM_HEADER_FILE_SEVEN))
//@Namespace("viennacl")
class viennacl extends Pointer with InfoMapper {
    def map(infoMap: InfoMap ) {
      infoMap.put(new
          Info("viennacl::matrix<double,viennacl::row_majorr,8>").pointerTypes("VCLMatrix_double_row_major_8"))
      infoMap.put(new
          Info("viennacl::vector<double>").pointerTypes("VCLVector_double"))
      infoMap.put(new
          Info("viennacl::vector<double,1>").pointerTypes("VCLVector_double_1"))
      infoMap.put(new
          Info("viennacl::vector<float>").pointerTypes("VCLVector_float"))
//      infoMap.put(new
//          Info("viennacl::vector<float>").pointerTypes("VCLVector_float"))
////      infoMap.put(new
//          Info("viennacl::vector<float>").pointerTypes("VCLNorm_2_double"))
//      infoMap.put(new
//          Info("viennacl::vector<float>").pointerTypes("VCLVector_float"))

    }

}

object viennacl {
  final val PLATFORM_HEADER_FILE_ONE = "viennacl/matrix.hpp"
  final val PLATFORM_HEADER_FILE_TWO = "viennacl/vector.hpp"
  final val PLATFORM_HEADER_FILE_THREE = "CL/cl.h"
  final val PLATFORM_HEADER_FILE_FOUR = "viennacl/scalar.hpp"
  final val PLATFORM_HEADER_FILE_FIVE = "viennacl/linalg/prod.hpp"
  final val PLATFORM_HEADER_FILE_SIX = "viennacl/forwards.h"
  final val PLATFORM_HEADER_FILE_SEVEN = "viennacl/linalg/norm_2.hpp"


  final val MEMORY_NOT_INITIALIZED = 0
  final val MAIN_MEMORY = 1
  final val OPENCL_MEMORY = 2
  final val CUDA_MEMORY =3

//  final val PLATFORM_HEADER_FILE_SIX = "CL/cl2.hpp" //fatal error: CL/cl2.hpp: not found on travis Debian build
//  final val PLATFORM_HEADER_FILE_SEVEN = "CL/cl.h"
}
//#include "viennacl/scalar.hpp"
//#include "viennacl/vector.hpp"
//#include "viennacl/matrix.hpp"
//#include "viennacl/linalg/prod.hpp"
//#include "viennacl/tools/random.hpp"
//#include "viennacl/tools/timer.hpp"

