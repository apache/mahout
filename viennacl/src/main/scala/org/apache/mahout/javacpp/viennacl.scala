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
package org.apache.mahout.javacpp

import org.apache.mahout.javacpp.presets.viennacl
import org.bytedeco.javacpp
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._

class viennacl extends org.apache.mahout.javacpp.presets.viennacl {
  Loader.load
//  allocate
//  @native def allocate(): Unit
  // MAHOUT_HOME/viennacl/target/classes/org/apache/mahout/javacpp/linux-x86_64/libjniviennacl.so

  // viennacl::vector<double>
  @Name(Array("vector<double>")) class VCLVector_double extends Pointer {

    Loader.load
    allocate
    @native def allocate(): Unit
    //@native def allocate(p: Pointer): Unit
    @native def size: Long
    @native def resize(size: Int)

    }
//    vector.hpp:971
//    @native def VCLVector_double(@Cast("NumericT *") ptr_to_mem, @Cast ("viennacl::memory_types") mem_type, size_type vec_size, vcl_size_t start, size_type stride)

  @Name(Array("vector<double,1>")) class VCLVector_double_1 (@Cast(Array("NumericT *")) val ptr_to_mem:  DoublePointer,
                                                             @Cast(Array("viennacl::memory_types")) val mem_ype: Int = viennacl.MAIN_MEMORY,
                                                             @Cast(Array("size_type")) val vec_size: Long,
                                                             @Cast(Array("size_type")) val start:  Long = 0,
                                                             @Cast(Array("size_type")) val stride: Long = 1 )  extends Pointer {

    allocate()
    allocate(vec_size)
    //allocate(ptr_to_mem)
    Loader.load

    @native def allocate(): Unit
    @native def allocate(size: Long): Unit
   // @native def allocate(p: Pointer): Unit
    @native def size: Long
    @native def resize(size: Int)
  }

//   // skip info map here
//  @Namespace("viennacl::linalg")
//   @native @Cast(Array("viennacl::scalar_expression<const viennacl::vector_base<double>"+
//    ",const viennacl::vector_base<double>,viennacl::op_norm_2> "+
//    "norm_2(viennacl::vector_base<double> const & v)")) def VCLNorm_2_double(@ByRef @Cast(Array("vector<double,1>")) vec: VCLVector_double_1) : Double

  @native @Cast(Array("viennacl::scalar_expression < const viennacl::vector_base<double>," +
    " const viennacl::vector_base<double>, viennacl::op_norm_2 > 	viennacl::linalg::norm_2")) def VCLNorm_2_double
                                                  (@ByRef @Cast(Array("vector<double,1>")) vec: VCLVector_double_1): Double


  // viennacl::vector<float>
  @Name(Array("vector<float>")) class VCLVector_float extends Pointer {

    Loader.load
    allocate

    @native def allocate(): Unit
    @native def size: Long
    @native def resize(size: Int)

  }



  // viennacl::matrix<double,row_major,8>
//  @Name(Array("matrix<double,viennacl::row_major,8>")) class VCLMatrix_double_row_major_8 extends Pointer {
//
//    Loader.load
//    allocate
//
//    @native def allocate(): Unit
//
//
//
//    //    @ByRef
////    @Name(Array("operator =")) def put(@Const @ByRef from: VCLMatrix_double_row_major_8): VCLMatrix_double_row_major_8
//
//    //    @native def size: Long
////    @native def resize(size: Int)
//
//
//  }

//  template<class NumericT, typename SizeT, typename DistanceT>
//  vector_base<NumericT, SizeT, DistanceT>::vector_base(NumericT * ptr_to_mem, viennacl::memory_types mem_type, size_type vec_size, vcl_size_t start, size_type stride)



  /** @brief Creates the matrix with the given dimensions
    * @param rows     Number of rows
    * @param columns  Number of columns
    * @param ctx      Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
    */
//  template<class NumericT, typename SizeT, typename DistanceT>
//  matrix_base<NumericT, SizeT, DistanceT>::matrix_base(size_type rows, size_type columns, bool is_row_major, viennacl::context ctx)
//    : size1_(rows), size2_(columns), start1_(0), start2_(0), stride1_(1), stride2_(1),
//  internal_size1_(viennacl::tools::align_to_multiple<size_type>(rows, dense_padding_size)),
//  internal_size2_(viennacl::tools::align_to_multiple<size_type>(columns, dense_padding_size)),
//  row_major_fixed_(true), row_major_(is_row_major)
////  {
//    if (rows > 0 && columns > 0)
//    {
//      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), ctx);
//      clear();
//    }
//  }

}


