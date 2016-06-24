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

package org.apache.mahout.math.scalabindings

import org.apache.log4j.Level

import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

import scala.math._

class MathSuite extends FunSuite with MahoutSuite {

  private final implicit val log = getLog(classOf[MathSuite])

  test("chol") {

    // try to solve Ax=b with cholesky:
    // this requires
    // (LL')x = B
    // L'x= (L^-1)B
    // x=(L'^-1)(L^-1)B

    val a = dense((1, 2, 3), (2, 3, 4), (3, 4, 5.5))

    // make sure it is symmetric for a valid solution
    a := a.t %*% a

    trace(s"A= \n$a")

    val b = dense((9, 8, 7)).t

    trace(s"b = \n$b")

    // Fails if chol(a, true)
    val ch = chol(a)

    trace(s"L = \n${ch.getL}")

    trace(s"(L^-1)b =\n${ch.solveLeft(b)}\n")

    val x = ch.solveRight(eye(3)) %*% ch.solveLeft(b)

    trace(s"x = \n$x")

    val axmb = (a %*% x) - b

    trace(s"AX - B = \n$axmb")

    axmb.norm should be < 1e-10

  }

  test("chol2") {

    val vtv = new DenseSymmetricMatrix(
      Array(
        0.0021401286568947376, 0.001309251254596442, 0.0016003218703045058,
        0.001545407014131058, 0.0012772546647977234,
        0.001747768702674435
      ), true)

    printf("V'V=\n%s\n", vtv cloned)

    val vblock = dense(
      (0.0012356809018514347, 0.006141139195280868, 8.037742467936037E-4),
      (0.007910767859830255, 0.007989899899005457, 0.006877961936587515),
      (0.007011211118759952, 0.007458865101641882, 0.0048344749320346795),
      (0.006578789899685284, 0.0010812485516549452, 0.0062146270886981655)
    )

    val d = diag(15.0, 4)


    val b = dense(
      0.36378319648203084,
      0.3627384439613304,
      0.2996934112658234)

    printf("B=\n%s\n", b)


    val cholArg = vtv + (vblock.t %*% d %*% vblock) + diag(4e-6, 3)

    printf("cholArg=\n%s\n", cholArg)

    printf("V'DV=\n%s\n", (vblock.t %*% d %*% vblock))

    printf("V'V+V'DV=\n%s\n", vtv + (vblock.t %*% d %*% vblock))

    val ch = chol(cholArg)

    printf("L=\n%s\n", ch.getL)

    val x = ch.solveRight(eye(cholArg.nrow)) %*% ch.solveLeft(b)

    printf("X=\n%s\n", x)

    assert((cholArg %*% x - b).norm < 1e-10)

  }

  test("qr") {
    val a = dense((1, 2, 3), (2, 3, 6), (3, 4, 5), (4, 7, 8))
    val (q, r) = qr(a)

    printf("Q=\n%s\n", q)
    printf("R=\n%s\n", r)

    for (i <- 0 until q.ncol; j <- i + 1 until q.ncol)
      assert(abs(q(::, i) dot q(::, j)) < 1e-10)
  }

  test("solve matrix-vector") {
    val a = dense((1, 3), (4, 2))
    val b = dvec(11, 14)
    val x = solve(a, b)

    val control = dvec(2, 3)

    (control - x).norm(2) should be < 1e-10
  }

  test("solve matrix-matrix") {
    val a = dense((1, 3), (4, 2))
    val b = dense(11, 14)
    val x = solve(a, b)

    val control = dense(2, 3)

    (control - x).norm should be < 1e-10
  }

  test("solve to obtain inverse") {
    val a = dense((1, 3), (4, 2))
    val x = solve(a)

    val identity = a %*% x

    val control = eye(identity.ncol)

    (control - identity).norm should be < 1e-10
  }

  test("solve rejects non-square matrix") {
    intercept[IllegalArgumentException] {
      val a = dense((1, 2, 3), (4, 5, 6))
      val b = dvec(1, 2)
      solve(a, b)
    }
  }

  test("solve rejects singular matrix") {
    intercept[IllegalArgumentException] {
      val a = dense((1, 2), (2 , 4))
      val b = dvec(1, 2)
      solve(a, b)
    }
  }

  test("svd") {

    val a = dense((1, 2, 3), (3, 4, 5))

    val (u, v, s) = svd(a)

    printf("U:\n%s\n", u.toString)
    printf("V:\n%s\n", v.toString)
    printf("Sigma:\n%s\n", s.toString)

    val aBar = u %*% diagv(s) %*% v.t

    val amab = a - aBar

    printf("A-USV'=\n%s\n", amab.toString)

    assert(amab.norm < 1e-10)

  }

  test("random uniform") {
    val omega1 = Matrices.symmetricUniformView(2, 3, 1234)
    val omega2 = Matrices.symmetricUniformView(2, 3, 1234)

    val a = sparse(
      0 -> 1 :: 1 -> 2 :: Nil,
      0 -> 3 :: 1 -> 4 :: Nil,
      0 -> 2 :: 1 -> 0.0 :: Nil
    )

    val block = a(0 to 0, ::).cloned
    val block2 = a(1 to 1, ::).cloned

    (block %*% omega1 - (a %*% omega2)(0 to 0, ::)).norm should be < 1e-7
    (block2 %*% omega1 - (a %*% omega2)(1 to 1, ::)).norm should be < 1e-7

  }

  test("sqDist(X,Y)") {
    val m = 100
    val n = 300
    val d = 7
    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5
    val mxY = Matrices.symmetricUniformView(n, d, 1234).cloned += 10

    val mxDsq = sqDist(mxX, mxY)
    val mxDsqControl = new DenseMatrix(m, n) := { (r, c, _) ⇒ (mxX(r, ::) - mxY(c, ::)) ^= 2 sum }
    (mxDsq - mxDsqControl).norm should be < 1e-7
  }

  test("sqDist(X)") {
    val m = 100
    val d = 7
    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5

    val mxDsq = sqDist(mxX)
    val mxDsqControl = sqDist(mxX, mxX)
    (mxDsq - mxDsqControl).norm should be < 1e-7
  }

  test("sparsity analysis") {
    setLogLevel(Level.DEBUG)

    val m = 500
    val n = 800
    val mxA = new DenseMatrix(m, n)

    densityAnalysis(mxA) shouldBe false
    densityAnalysis(mxA, .5) shouldBe false
    densityAnalysis(mxA + 1) shouldBe true
    densityAnalysis(mxA + 1, .95) shouldBe true

    for (i ← 0 until m by 5) mxA(i, ::) := 1
    info(s"20% detected as dense?:${densityAnalysis(mxA)}")
    mxA := 0

    for (i ← 0 until m by 3) mxA(i, ::) := 1
    info(s"33% detected as dense?:${densityAnalysis(mxA)}")
    mxA := 0

    for (i ← 0 until m by 4) mxA(i, ::) := 1
    info(s"25% detected as dense?:${densityAnalysis(mxA)}")

    for (i ← 0 until m by 2) mxA(i, ::) := 1
    info(s"50% detected as dense?:${densityAnalysis(mxA)}")

  }

}
