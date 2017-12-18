/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.algorithms

import org.scalatest.FunSuite
import org.apache.mahout.math.algorithms.neuralnet.Converters._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.mahout.test.MahoutSuite

class VectorConverterSuite extends FunSuite with MahoutSuite {

  val m1 = dense((1,2,3), (4,5,6))
  val m3 = dense((7), (8), (9))

  val mA = Array(m1, m3)

  test("matrix2vec") {
    val v = flattenMatrixToVec(m1)
    val m2 = recomposeMatrixFromVec(v, m1.nrow, m1.ncol)
    (m1 - m2).norm should equal (0)
  }

  test("matrixArray2Vec") {
    val v = flattenMatrixArrayToVector(mA)
    val mA2 = recomposeMatrixArrayFromVec(v, Array( (2, 3), (3, 1)))
    mA.map(m => m.norm).deep should equal (mA2.map(m => m.norm).deep)
  }


  test("arrayOfMatrixArraysToVec") {


    val A = dense( (1,2,3), (4,5,6))
    val B = dense( (7,8), (9, 10) )

    val ama: Array[Array[_ <: Matrix]] = Array(Array(A), Array(B))

    val v = flattenArrayOfMatrixArraysToVector(ama)

    val sa = Array(Array((3,2)), Array((2,2)))
    val ama2 = recomposeArrayOfMatrixArraysFromVec(v, sa)

    ama.map(a => a.map(m => m.norm)).deep should equal (ama2.map(a => a.map(m => m.norm)).deep)
  }

}
