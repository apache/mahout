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

import org.apache.log4j.{Level, BasicConfigurator}
import org.scalatest.FunSuite
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.test.MahoutSuite

import org.apache.mahout.logging._

class RLikeVectorOpsSuite extends FunSuite with MahoutSuite {

  BasicConfigurator.configure()
  private[scalabindings] final implicit val log = getLog(classOf[RLikeVectorOpsSuite])
  setLogLevel(Level.DEBUG)

  test("Hadamard") {
    val a: Vector = (1, 2, 3)
    val b = (3, 4, 5)

    val c = a * b
    println(c)
    assert(c ===(3, 8, 15))
  }

  test("dot-view performance") {

    val dv1 = new DenseVector(500) := Matrices.uniformView(1, 500, 1234)(0, ::)
    val dv2 = new DenseVector(500) := Matrices.uniformView(1, 500, 1244)(0, ::)

    val nit = 300000

    // warm up
    dv1 dot dv2

    val dmsStart = System.currentTimeMillis()
    for (i ← 0 until nit)
      dv1 dot dv2
    val dmsMs = System.currentTimeMillis() - dmsStart

    val (dvv1, dvv2) = dv1(0 until dv1.length) → dv2(0 until dv2.length)

    // Warm up.
    dvv1 dot dvv2

    val dvmsStart = System.currentTimeMillis()
    for (i ← 0 until nit)
      dvv1 dot dvv2
    val dvmsMs = System.currentTimeMillis() - dvmsStart

    debug(f"dense vector dots:${dmsMs}%.2f ms.")
    debug(f"dense view dots:${dvmsMs}%.2f ms.")

  }

}
