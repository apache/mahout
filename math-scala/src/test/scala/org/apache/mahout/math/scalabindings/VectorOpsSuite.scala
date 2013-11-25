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

package org.apache.mahout.math.scalabindings

import org.scalatest.FunSuite
import org.apache.mahout.math.{RandomAccessSparseVector, Vector}
import MatrixOps._

/**
 * Created with IntelliJ IDEA.
 * User: dmitriy
 * Date: 6/21/13
 * Time: 10:26 PM
 * To change this template use File | Settings | File Templates.
 */
class VectorOpsSuite extends FunSuite {

  test("inline create") {

    val sparseVec = svec((5 -> 1) :: (10 -> 2.0) :: Nil)
    println(sparseVec)

    val sparseVec2: Vector = (5 -> 1.0) :: (10 -> 2.0) :: Nil
    println(sparseVec2)

    val sparseVec3: Vector = new RandomAccessSparseVector(100) := (5 -> 1.0) :: Nil
    println(sparseVec3)

    val denseVec1: Vector = (1.0, 1.1, 1.2)
    println(denseVec1)

    val denseVec2 = dvec(1, 0, 1.1, 1.2)
    println(denseVec2)
  }

  test("plus minus") {

    val a: Vector = (1, 2, 3)
    val b: Vector = (0 -> 3) :: (1 -> 4) :: (2 -> 5) :: Nil

    val c = a + b
    val d = b - a
    val e = -b - a

    assert(c ===(4, 6, 8))
    assert(d ===(2, 2, 2))
    assert(e ===(-4, -6, -8))

  }

  test("dot") {

    val a: Vector = (1, 2, 3)
    val b = (3, 4, 5)

    val c = a dot b
    println(c)
    assert(c == 26)

  }

}
