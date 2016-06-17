package org.apache.mahout.viennacl
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


import java.nio.{ByteBuffer, DoubleBuffer}

//import org.apache.mahout.javacpp.presets.viennacl
import org.apache.mahout.javacpp.viennacl
import org.apache.mahout.math.DenseVector
import org.bytedeco.javacpp.DoublePointer
import org.scalatest.{FunSuite, Matchers}

import org.apache.mahout.math._
import drm._
import scalabindings._
import RLikeOps._

class HelloVCLTestSuite extends FunSuite with Matchers {

  class VclCtx extends DistributedContext{
    val engine: DistributedEngine = null

    def close() {
    }
  }

  // Distributed Context to check for VCL Capabilities
  val vclCtx = new VclCtx()


  test("HelloVCLVector_double"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

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
    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }

  test("HelloVCLVector_float") {
    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      // create a new viennacl class based on CAFFE templata
      val vcl = new viennacl()

      // create a new vienna::vector<float>
      val nDVec = new vcl.VCLVector_float()

      // resize to 10 elements
      // vienna::vector<NumericT>::resize(int size)
      nDVec.resize(10)

      // ensure that the sies is 10 elements
      // vienna::vector<NumericT>::size()
      assert(nDVec.size == 10)
    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }
  test("Simple dense native vector from colt vector"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      // create a new viennacl class based on CAFFE templata
      val vcl = new viennacl()

      // create a JVM-backed vector
      val coltVec = new DenseVector(5)

      // add a few values
      coltVec(1)=1.0
      coltVec(2)=2.0
      coltVec(3)=3.0
      coltVec(4)=4.0
      coltVec(5)=5.0


      // get the pointer to the Backing Array
      val coltVecArray = coltVec.getBackingDataStructure


      // create a new vienna::vector<double,1>
      val vclVec = new vcl.VCLVector_double_1(new DoublePointer(DoubleBuffer.wrap(coltVecArray)),0 , 20, 0 ,1 )

      // ensure that both vclVec and coltVec are the same size
      // vienna::vector<NumericT>::size()
      assert(vclVec.size == coltVec.size())

      // check the L2 norms of each vector
      assert(vcl.VCLNorm_2_double(vclVec) == coltVec.norm(2))

    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }
}

