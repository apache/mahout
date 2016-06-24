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
import java.util.Random

import org.apache.mahout.javacpp.linalg.vcl_blas3._
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



  test("Simple dense %*% dense native mmul"){

    // probe to see if VCL libraries are installed
    if (vclCtx.useVCL) {

      val r1 = new Random(1234)

      val mxA:DenseMatrix = dense(1000,1000)
      val mxB:DenseMatrix = dense(1000,1000)
      // add some data
      mxA ::= {(_,_,v) => r1.nextDouble()}
      mxB ::= {(_,_,v) => r1.nextDouble()}

      val mxC = mxB.like()

      val mmulControl = mxA %*% mxB


      dense_dense_mmul(new DoublePointer(DoubleBuffer.wrap(mxA.getBackingArray)) )


    } else {
      printf("No Native VCL library found... Skipping test")
    }
  }
}

