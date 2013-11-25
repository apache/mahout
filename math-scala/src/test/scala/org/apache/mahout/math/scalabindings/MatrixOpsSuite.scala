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
import MatrixOps._
import scala._


class MatrixOpsSuite extends FunSuite {


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

    println(a.toString)
    println(a.sum)

  }

  test("assignments") {

    val a = dense((1, 2, 3), (3, 4, 5))

    val b = a cloned

    b(0, 0) = 2.0

    printf("B=\n%s\n", b)

    assert((b - a).norm - 1 < 1e-10)

    val e = eye(5)

    printf("I(5)=\n%s\n", e)

    a(0 to 1, 1 to 2) = dense((3, 2), (2, 3))
    a(0 to 1, 1 to 2) := dense((3, 2), (2, 3))


  }



  test("sparse") {

    val a = sparse((1, 3) :: Nil,
      (0, 2) ::(1, 2.5) :: Nil
    )
    println(a.toString)
  }

}