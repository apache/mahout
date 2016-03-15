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

import org.scalatest.{Matchers, FunSuite}
import RLikeOps._
import scala._
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.math.{RandomAccessSparseVector, SequentialAccessSparseVector, Matrices}
import org.apache.mahout.common.RandomUtils

import scala.util.Random


class MatrixOpsSuite extends FunSuite with MahoutSuite {

  test("equivalence") {
    val a = dense((1, 2, 3), (3, 4, 5))
    val b = dense((1, 2, 3), (3, 4, 5))
    val c = dense((1, 4, 3), (3, 4, 5))
    assert(a === b)
    assert(a !== c)
  }

  test("elementwise plus, minus") {
    val a = dense((1, 2, 3), (3, 4, 5))
    val b = dense((1, 1, 2), (2, 1, 1))

    val c = a + b
    assert(c(0, 0) == 2)
    assert(c(1, 2) == 6)
    println(c.toString)
  }

  test("matrix, vector slicing") {

    val a = dense((1, 2, 3), (3, 4, 5))

    assert(a(::, 0).sum == 4)
    assert(a(1, ::).sum == 12)

    assert(a(0 to 1, 1 to 2).sum == 14)

    // assign to slice-vector
    a(0, 0 to 1) :=(3, 5)
    // or
    a(0, 0 to 1) = (3, 5)

    assert(a(0, ::).sum == 11)

    println(a.toString)

    // assign to a slice-matrix
    a(0 to 1, 0 to 1) := dense((1, 1), (2, 2.5))

    // or
    a(0 to 1, 0 to 1) = dense((1, 1), (2, 2.5))

    println(a)
    println(a.sum)

    val b = dense((1, 2, 3), (3, 4, 5))
    b(0, ::) -= dvec(1, 2, 3)
    println(b)
    b(0, ::) should equal(dvec(0, 0, 0))

  }

  test("assignments") {

    val a = dense((1, 2, 3), (3, 4, 5))

    val b = a cloned

    b(0, 0) = 2.0

    printf("B=\n%s\n", b)

    assert((b - a).norm - 1 < 1e-10)

    val e = eye(5)

    println(s"I(5)=\n$e")

    a(0 to 1, 1 to 2) = dense((3, 2), (2, 3))
    a(0 to 1, 1 to 2) := dense((3, 2), (2, 3))

    println(s"a=$a")

    a(0 to 1, 1 to 2) := { _ => 45}
    println(s"a=$a")

//    a(0 to 1, 1 to 2) ::= { _ => 44}
    println(s"a=$a")

    // Sparse assignment to a sparse block
    val c = sparse(0 -> 1 :: Nil, 2 -> 2 :: Nil, 1 -> 5 :: Nil)
    val d = c.cloned

    println(s"d=$d")
    d.ncol shouldBe 3

    d(::, 1 to 2) ::= { _ => 4}
    println(s"d=$d")
    d(::, 1 to 2).sum shouldBe 8

    d ::= {_ => 5}
    d.sum shouldBe 15

    val f = c.cloned.t
    f ::= {_ => 6}
    f.sum shouldBe 18

    val g = c.cloned
    g(::, 1 until g.nrow) ::= { x => if (x <= 0) 0.0 else 1.0}
    g.sum shouldBe 3
  }

  test("functional apply()") {
    val mxA = sparse (
      (1 -> 3) :: (7 -> 7) :: Nil,
      (4 -> 5) :: (5 -> 8) :: Nil
    )
    val mxAControl = mxA cloned

    (mxA(x ⇒ x + 1) - (mxAControl + 1)).norm should be < 1e-7
    (mxA(x ⇒ x * 2) - (2 * mxAControl)).norm should be < 1e-7

  }

  test("sparse") {

    val a = sparse((1, 3) :: Nil,
      (0, 2) ::(1, 2.5) :: Nil
    )
    println(a.toString)
  }

  test("colSums, rowSums, colMeans, rowMeans, numNonZeroElementsPerColumn") {
    val a = dense(
      (2, 3, 4),
      (3, 4, 5)
    )

    a.colSums() should equal(dvec(5, 7, 9))
    a.rowSums() should equal(dvec(9, 12))
    a.colMeans() should equal(dvec(2.5, 3.5, 4.5))
    a.rowMeans() should equal(dvec(3, 4))
    a.numNonZeroElementsPerColumn() should equal(dvec(2,2,2))
    a.numNonZeroElementsPerRow() should equal(dvec(3,3))

  }

  test("numNonZeroElementsPerColumn and Row") {
    val a = dense(
      (2, 3, 4),
      (3, 4, 5),
      (-5, 0, -1),
      (0, 0, 1)
    )

    a.numNonZeroElementsPerColumn() should equal(dvec(3,2,4))
    a.numNonZeroElementsPerRow() should equal(dvec(3,3,2,1))
  }

  test("Vector Assignment performance") {

    val n = 1000
    val k = (n * 0.1).toInt
    val nIters = 10000

    val rnd = RandomUtils.getRandom

    val src = new SequentialAccessSparseVector(n)
    for (i <- 0 until k) src(rnd.nextInt(n)) = rnd.nextDouble()

    val times = (0 until 50).map { i =>
      val ms = System.currentTimeMillis()
      var j = 0
      while (j < nIters) {
        new SequentialAccessSparseVector(n) := src
        j += 1
      }
      System.currentTimeMillis() - ms
    }

        .tail

    val avgTime = times.sum.toDouble / times.size

    printf("Average assignment seqSparse2seqSparse time: %.3f ms\n", avgTime)

    val times2 = (0 until 50).map { i =>
      val ms = System.currentTimeMillis()
      var j = 0
      while (j < nIters) {
        new SequentialAccessSparseVector(n) := (new RandomAccessSparseVector(n) := src)
        j += 1
      }
      System.currentTimeMillis() - ms
    }

        .tail

    val avgTime2 = times2.sum.toDouble / times2.size

    printf("Average assignment seqSparse2seqSparse via Random Access Sparse time: %.3f ms\n", avgTime2)

  }



}