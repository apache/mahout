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

@Platform(include=Array(viennacl.PLATFORM_HEADER_FILE_ONE,viennacl.PLATFORM_HEADER_FILE_TWO))
//@Namespace("viennacl")
class viennacl extends Pointer with InfoMapper {
    def map(infoMap: InfoMap ) {
      infoMap.put(new
          Info("viennacl::matrix<double,row_major,8>").pointerTypes("DoubleRowmajorMatrixClass8"))
      infoMap.put(new
          Info("viennacl::vector<double>").pointerTypes("VCLVector_double"))
      infoMap.put(new
          Info("viennacl::vector<float>").pointerTypes("VCLVector_float"))

    }



}

object viennacl {
  final val PLATFORM_HEADER_FILE_ONE = "matrix.hpp"
  final val PLATFORM_HEADER_FILE_TWO = "vector.hpp"
}
