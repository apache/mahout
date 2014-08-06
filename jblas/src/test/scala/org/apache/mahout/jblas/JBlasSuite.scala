/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.jblas

import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.DenseMatrix
import scala.util.Random

class JBlasSuite extends FunSuite with Matchers {

  def time(f: => Unit) = {
    val s = System.currentTimeMillis
    f
    System.currentTimeMillis - s
  }

  test("matrix multiplication") {
    val rng = new Random
    val sizeX = 1024
    val sizeY = 1024
    val controlA = new DenseMatrix(sizeX, sizeY) assign { rng.nextDouble * 100 }
    val controlB = new DenseMatrix(sizeY, sizeX) assign { rng.nextDouble * 100 }
    val A = new JBlasMatrix(sizeX, sizeY) assign controlA
    val B = new JBlasMatrix(sizeY, sizeX) assign controlB

    var controlAB: Matrix = null
    var AB: Matrix = null

    println("Normal multiplication (ms) = " + time {
      controlAB = controlA.times(controlB)
    })

    println("jBLAS multiplication (ms) = " + time {
      AB = A.times(B)
    })

    AB.minus(controlAB).zSum should be < 1e-10
  }
}
