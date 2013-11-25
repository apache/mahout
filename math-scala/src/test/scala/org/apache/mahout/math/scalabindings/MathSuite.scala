package org.apache.mahout.math.scalabindings

import org.scalatest.FunSuite
import org.apache.mahout.math.DenseSymmetricMatrix
import scala.math._
import RLikeOps._
import scala._

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

class MathSuite extends FunSuite {

  test("chol") {

    // try to solve Ax=b with cholesky:
    // this requires
    // (LL')x = B
    // L'x= (L^-1)B
    // x=(L'^-1)(L^-1)B

    val a = dense((1, 2, 3), (2, 3, 4), (3, 4, 5.5))

    // make sure it is symmetric for a valid solution
    a := a.t %*% a

    printf("A= \n%s\n", a)

    val b = dense((9, 8, 7)).t

    printf("b = \n%s\n", b)

    // fails if chol(a,true)
    val ch = chol(a)

    printf("L = \n%s\n", ch.getL)

    printf("(L^-1)b =\n%s\n", ch.solveLeft(b))

    val x = ch.solveRight(eye(3)) %*% ch.solveLeft(b)

    printf("x = \n%s\n", x.toString)

    val axmb = (a %*% x) - b

    printf("AX - B = \n%s\n", axmb.toString)

    assert(axmb.norm < 1e-10)

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
      (0.36378319648203084),
      (0.3627384439613304),
      (0.2996934112658234))

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

  test("ssvd") {

    val a = dense((1, 2, 3), (3, 4, 5))

    val (u, v, s) = ssvd(a, 2, q = 1)

    printf("U:\n%s\n", u)
    printf("V:\n%s\n", v)
    printf("Sigma:\n%s\n", s)

    val aBar = u %*% diagv(s) %*% v.t

    val amab = a - aBar

    printf("A-USV'=\n%s\n", amab)

    assert(amab.norm < 1e-10)

  }

}
