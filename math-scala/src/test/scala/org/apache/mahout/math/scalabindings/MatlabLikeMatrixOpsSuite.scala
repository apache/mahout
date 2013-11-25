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
import MatlabLikeOps._
import scala.Predef._

class MatlabLikeMatrixOpsSuite extends FunSuite {

  test("multiplication") {

    val a = dense((1, 2, 3), (3, 4, 5))
    val b = dense(1, 4, 5)
    val m = a * b

    assert(m(0, 0) == 24)
    assert(m(1, 0) == 44)
    println(m.toString)
  }

  test("Hadamard") {
    val a = dense(
      (1, 2, 3),
      (3, 4, 5)
    )
    val b = dense(
      (1, 1, 2),
      (2, 1, 1)
    )

    val c = a *@ b

    printf("C=\n%s\n", c)

    assert(c(0, 0) == 1)
    assert(c(1, 2) == 5)
    println(c.toString)

    val d = a *@ 5.0
    assert(d(0, 0) == 5)
    assert(d(1, 1) == 20)

    a *@= b
    assert(a(0, 0) == 1)
    assert(a(1, 2) == 5)
    println(a.toString)

  }

}
