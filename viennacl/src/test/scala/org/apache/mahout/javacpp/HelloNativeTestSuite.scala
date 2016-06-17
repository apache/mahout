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

import org.apache.mahout.javacpp.viennacl
import org.scalatest.{FunSuite, Matchers}


class HelloNativeTestSuite extends FunSuite with Matchers {

  test("HelloVCLVector_double"){
    // create a new viennacl class based on CAFFE templata
    val vcl = new viennacl()

    // create a new vienna::vector<double>
    val nDVec = new vcl.VCLVector_double()

    // resize to 10 elements
    // vienna::vector<NumericT>::resize(int size)
    nDVec.resize(10)

    // ensure that the sies is 10 elements
    // vienna::vector<NumericT>::size()
    assert(nDVec.size == 10)
  }

  test("HelloVCLVector_float"){

    // create a new viennacl class based on CAFFE template
    val vcl = new viennacl()

    // create a new vienna::vector<float>
    val nDVec = new vcl.VCLVector_float()

    // resize to 10 elements
    // vienna::vector<NumericT>::resize(int size)
    nDVec.resize(10)

    // ensure that the sies is 10 elements
    // vienna::vector<NumericT>::size()
    assert(nDVec.size == 10)
  }
//  test("VCLMatrix_double_row_major_8"){
//
//    // create a new viennacl class based on CAFFE templata
//    val vcl = new viennacl()
//
//    // create a new vienna::vector<float>
//    val vclnDMxa = new vcl.VCLMatrix_double_row_major_8(10,10)
//
//    // resize to 10 elements
//    // vienna::vector<NumericT>::resize(int size)
////    vclnDMxa(10,10)
////
////    // ensure that the sies is 10 elements
////    // vienna::vector<NumericT>::size()
////    assert(vclnDMxa.size == 10)
//  }

}
