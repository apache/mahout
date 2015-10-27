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

import java.util

import org.apache.log4j.Level
import org.apache.mahout.math._
import org.scalatest.FunSuite
import RLikeOps._
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.logging._
import scala.collection.JavaConversions._
import scala.util.Random

class RLikeMatrixOpsSuite extends FunSuite with MahoutSuite {

  test("multiplication") {

    val a = dense((1, 2, 3), (3, 4, 5))
    val b = dense(1, 4, 5)
    val m = a %*% b

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

    val c = a * b

    printf("C=\n%s\n", c)

    assert(c(0, 0) == 1)
    assert(c(1, 2) == 5)
    println(c.toString)

    val d = a * 5.0
    assert(d(0, 0) == 5)
    assert(d(1, 1) == 20)

    a *= b
    assert(a(0, 0) == 1)
    assert(a(1, 2) == 5)
    println(a.toString)

  }

  test("Uniform view") {
    val mxUnif = Matrices.symmetricUniformView(5000000, 5000000, 1234)
  }

  /** Test dsl overloads over scala operations over matrices */
  test ("scalarOps") {
    val a = dense(
      (1, 2, 3),
      (3, 4, 5)
    )

    (10 * a - (10 *: a)).norm shouldBe 0
    (10 + a - (10 +: a)).norm shouldBe 0
    (10 - a - (10 -: a)).norm shouldBe 0
    (10 / a - (10 /: a)).norm shouldBe 0

  }

  test("Multiplication experimental performance") {

    getLog(MMul.getClass).setLevel(Level.DEBUG)

    val d = 300
    val n = 3

    // Dense row-wise
    val mxAd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) + 1
    val mxBd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) - 1

    val rnd = new Random(1234)

    // Sparse rows
    val mxAsr = (new SparseRowMatrix(d,
      d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() + 1 else 0.0 }) cloned
    val mxBsr = (new SparseRowMatrix(d,
      d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() - 1 else 0.0 }) cloned

    // Hanging sparse rows
    val mxAs = (new SparseMatrix(d, d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() + 1 else 0.0 }) cloned
    val mxBs = (new SparseMatrix(d, d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() - 1 else 0.0 }) cloned

    // DIAGONAL
    val mxD = diagv(dvec(Array.tabulate(d)(_ => rnd.nextGaussian())))

    def time(op: => Unit): Long = {
      val ms = System.currentTimeMillis()
      op
      System.currentTimeMillis() - ms
    }

    def getMmulAvgs(mxA: Matrix, mxB: Matrix, n: Int) = {

      var control: Matrix = null
      var mmulVal: Matrix = null

      val current = Stream.range(0, n).map { _ => time {control = mxA.times(mxB)} }.sum.toDouble / n
      val experimental = Stream.range(0, n).map { _ => time {mmulVal = MMul(mxA, mxB, None)} }.sum.toDouble / n
      (control - mmulVal).norm should be < 1e-10
      current -> experimental
    }

    // Dense matrix tests.
    println(s"Ad %*% Bd: ${getMmulAvgs(mxAd, mxBd, n)}")
    println(s"Ad(::,::) %*% Bd: ${getMmulAvgs(mxAd(0 until mxAd.nrow,::), mxBd, n)}")
    println(s"Ad' %*% Bd: ${getMmulAvgs(mxAd.t, mxBd, n)}")
    println(s"Ad %*% Bd': ${getMmulAvgs(mxAd, mxBd.t, n)}")
    println(s"Ad' %*% Bd': ${getMmulAvgs(mxAd.t, mxBd.t, n)}")
    println(s"Ad'' %*% Bd'': ${getMmulAvgs(mxAd.t.t, mxBd.t.t, n)}")
    println

    // Sparse row matrix tests.
    println(s"Asr %*% Bsr: ${getMmulAvgs(mxAsr, mxBsr, n)}")
    println(s"Asr' %*% Bsr: ${getMmulAvgs(mxAsr.t, mxBsr, n)}")
    println(s"Asr %*% Bsr': ${getMmulAvgs(mxAsr, mxBsr.t, n)}")
    println(s"Asr' %*% Bsr': ${getMmulAvgs(mxAsr.t, mxBsr.t, n)}")
    println(s"Asr'' %*% Bsr'': ${getMmulAvgs(mxAsr.t.t, mxBsr.t.t, n)}")
    println

    // Sparse matrix tests.
    println(s"Asm %*% Bsm: ${getMmulAvgs(mxAs, mxBs, n)}")
    println(s"Asm' %*% Bsm: ${getMmulAvgs(mxAs.t, mxBs, n)}")
    println(s"Asm %*% Bsm': ${getMmulAvgs(mxAs, mxBs.t, n)}")
    println(s"Asm' %*% Bsm': ${getMmulAvgs(mxAs.t, mxBs.t, n)}")
    println(s"Asm'' %*% Bsm'': ${getMmulAvgs(mxAs.t.t, mxBs.t.t, n)}")
    println

    // Mixed sparse matrix tests.
    println(s"Asm %*% Bsr: ${getMmulAvgs(mxAs, mxBsr, n)}")
    println(s"Asm' %*% Bsr: ${getMmulAvgs(mxAs.t, mxBsr, n)}")
    println(s"Asm %*% Bsr': ${getMmulAvgs(mxAs, mxBsr.t, n)}")
    println(s"Asm' %*% Bsr': ${getMmulAvgs(mxAs.t, mxBsr.t, n)}")
    println(s"Asm'' %*% Bsr'': ${getMmulAvgs(mxAs.t.t, mxBsr.t.t, n)}")
    println

    println(s"Asr %*% Bsm: ${getMmulAvgs(mxAsr, mxBs, n)}")
    println(s"Asr' %*% Bsm: ${getMmulAvgs(mxAsr.t, mxBs, n)}")
    println(s"Asr %*% Bsm': ${getMmulAvgs(mxAsr, mxBs.t, n)}")
    println(s"Asr' %*% Bsm': ${getMmulAvgs(mxAsr.t, mxBs.t, n)}")
    println(s"Asr'' %*% Bsm'': ${getMmulAvgs(mxAsr.t.t, mxBs.t.t, n)}")
    println

    // Mixed dense/sparse
    println(s"Ad %*% Bsr: ${getMmulAvgs(mxAd, mxBsr, n)}")
    println(s"Ad' %*% Bsr: ${getMmulAvgs(mxAd.t, mxBsr, n)}")
    println(s"Ad %*% Bsr': ${getMmulAvgs(mxAd, mxBsr.t, n)}")
    println(s"Ad' %*% Bsr': ${getMmulAvgs(mxAd.t, mxBsr.t, n)}")
    println(s"Ad'' %*% Bsr'': ${getMmulAvgs(mxAd.t.t, mxBsr.t.t, n)}")
    println

    println(s"Asr %*% Bd: ${getMmulAvgs(mxAsr, mxBd, n)}")
    println(s"Asr' %*% Bd: ${getMmulAvgs(mxAsr.t, mxBd, n)}")
    println(s"Asr %*% Bd': ${getMmulAvgs(mxAsr, mxBd.t, n)}")
    println(s"Asr' %*% Bd': ${getMmulAvgs(mxAsr.t, mxBd.t, n)}")
    println(s"Asr'' %*% Bd'': ${getMmulAvgs(mxAsr.t.t, mxBd.t.t, n)}")
    println

    println(s"Ad %*% Bsm: ${getMmulAvgs(mxAd, mxBs, n)}")
    println(s"Ad' %*% Bsm: ${getMmulAvgs(mxAd.t, mxBs, n)}")
    println(s"Ad %*% Bsm': ${getMmulAvgs(mxAd, mxBs.t, n)}")
    println(s"Ad' %*% Bsm': ${getMmulAvgs(mxAd.t, mxBs.t, n)}")
    println(s"Ad'' %*% Bsm'': ${getMmulAvgs(mxAd.t.t, mxBs.t.t, n)}")
    println

    println(s"Asm %*% Bd: ${getMmulAvgs(mxAs, mxBd, n)}")
    println(s"Asm' %*% Bd: ${getMmulAvgs(mxAs.t, mxBd, n)}")
    println(s"Asm %*% Bd': ${getMmulAvgs(mxAs, mxBd.t, n)}")
    println(s"Asm' %*% Bd': ${getMmulAvgs(mxAs.t, mxBd.t, n)}")
    println(s"Asm'' %*% Bd'': ${getMmulAvgs(mxAs.t.t, mxBd.t.t, n)}")
    println

    // Diagonal cases
    println(s"Ad %*% D: ${getMmulAvgs(mxAd, mxD, n)}")
    println(s"Asr %*% D: ${getMmulAvgs(mxAsr, mxD, n)}")
    println(s"Asm %*% D: ${getMmulAvgs(mxAs, mxD, n)}")
    println(s"D %*% Ad: ${getMmulAvgs(mxD, mxAd, n)}")
    println(s"D %*% Asr: ${getMmulAvgs(mxD, mxAsr, n)}")
    println(s"D %*% Asm: ${getMmulAvgs(mxD, mxAs, n)}")
    println

    println(s"Ad' %*% D: ${getMmulAvgs(mxAd.t, mxD, n)}")
    println(s"Asr' %*% D: ${getMmulAvgs(mxAsr.t, mxD, n)}")
    println(s"Asm' %*% D: ${getMmulAvgs(mxAs.t, mxD, n)}")
    println(s"D %*% Ad': ${getMmulAvgs(mxD, mxAd.t, n)}")
    println(s"D %*% Asr': ${getMmulAvgs(mxD, mxAsr.t, n)}")
    println(s"D %*% Asm': ${getMmulAvgs(mxD, mxAs.t, n)}")
    println

    // Self-squared cases
    println(s"Ad %*% Ad': ${getMmulAvgs(mxAd, mxAd.t, n)}")
    println(s"Ad' %*% Ad: ${getMmulAvgs(mxAd.t, mxAd, n)}")
    println(s"Ad' %*% Ad'': ${getMmulAvgs(mxAd.t, mxAd.t.t, n)}")
    println(s"Ad'' %*% Ad': ${getMmulAvgs(mxAd.t.t, mxAd.t, n)}")

  }


  test("elementwise experimental performance") {

    val d = 500
    val n = 3

    // Dense row-wise
    val mxAd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) + 1
    val mxBd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) - 1

    val rnd = new Random(1234)

    // Sparse rows
    val mxAsr = (new SparseRowMatrix(d,
      d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() + 1 else 0.0 }) cloned
    val mxBsr = (new SparseRowMatrix(d,
      d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() - 1 else 0.0 }) cloned

    // Hanging sparse rows
    val mxAs = (new SparseMatrix(d, d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() + 1 else 0.0 }) cloned
    val mxBs = (new SparseMatrix(d, d) := { _ => if (rnd.nextDouble() < 0.1) rnd.nextGaussian() - 1 else 0.0 }) cloned

    // DIAGONAL
    val mxD = diagv(dvec(Array.tabulate(d)(_ => rnd.nextGaussian())))

    def time(op: => Unit): Long = {
      val ms = System.currentTimeMillis()
      op
      System.currentTimeMillis() - ms
    }

    def getEWAvgs(mxA: Matrix, mxB: Matrix, n: Int) = {

      var control: Matrix = null
      var mmulVal: Matrix = null

      val current = Stream.range(0, n).map { _ => time {control = mxA + mxB} }.sum.toDouble / n
      val experimental = Stream.range(0, n).map { _ => time {mmulVal = mxA + mxB} }.sum.toDouble / n
      (control - mmulVal).norm should be < 1e-10
      current -> experimental
    }

    // Dense matrix tests.
    println(s"Ad + Bd: ${getEWAvgs(mxAd, mxBd, n)}")
    println(s"Ad' + Bd: ${getEWAvgs(mxAd.t, mxBd, n)}")
    println(s"Ad + Bd': ${getEWAvgs(mxAd, mxBd.t, n)}")
    println(s"Ad' + Bd': ${getEWAvgs(mxAd.t, mxBd.t, n)}")
    println(s"Ad'' + Bd'': ${getEWAvgs(mxAd.t.t, mxBd.t.t, n)}")
    println

    // Sparse row matrix tests.
    println(s"Asr + Bsr: ${getEWAvgs(mxAsr, mxBsr, n)}")
    println(s"Asr' + Bsr: ${getEWAvgs(mxAsr.t, mxBsr, n)}")
    println(s"Asr + Bsr': ${getEWAvgs(mxAsr, mxBsr.t, n)}")
    println(s"Asr' + Bsr': ${getEWAvgs(mxAsr.t, mxBsr.t, n)}")
    println(s"Asr'' + Bsr'': ${getEWAvgs(mxAsr.t.t, mxBsr.t.t, n)}")
    println

    // Sparse matrix tests.
    println(s"Asm + Bsm: ${getEWAvgs(mxAs, mxBs, n)}")
    println(s"Asm' + Bsm: ${getEWAvgs(mxAs.t, mxBs, n)}")
    println(s"Asm + Bsm': ${getEWAvgs(mxAs, mxBs.t, n)}")
    println(s"Asm' + Bsm': ${getEWAvgs(mxAs.t, mxBs.t, n)}")
    println(s"Asm'' + Bsm'': ${getEWAvgs(mxAs.t.t, mxBs.t.t, n)}")
    println

    // Mixed sparse matrix tests.
    println(s"Asm + Bsr: ${getEWAvgs(mxAs, mxBsr, n)}")
    println(s"Asm' + Bsr: ${getEWAvgs(mxAs.t, mxBsr, n)}")
    println(s"Asm + Bsr': ${getEWAvgs(mxAs, mxBsr.t, n)}")
    println(s"Asm' + Bsr': ${getEWAvgs(mxAs.t, mxBsr.t, n)}")
    println(s"Asm'' + Bsr'': ${getEWAvgs(mxAs.t.t, mxBsr.t.t, n)}")
    println

    println(s"Asr + Bsm: ${getEWAvgs(mxAsr, mxBs, n)}")
    println(s"Asr' + Bsm: ${getEWAvgs(mxAsr.t, mxBs, n)}")
    println(s"Asr + Bsm': ${getEWAvgs(mxAsr, mxBs.t, n)}")
    println(s"Asr' + Bsm': ${getEWAvgs(mxAsr.t, mxBs.t, n)}")
    println(s"Asr'' + Bsm'': ${getEWAvgs(mxAsr.t.t, mxBs.t.t, n)}")
    println

    // Mixed dense/sparse
    println(s"Ad + Bsr: ${getEWAvgs(mxAd, mxBsr, n)}")
    println(s"Ad' + Bsr: ${getEWAvgs(mxAd.t, mxBsr, n)}")
    println(s"Ad + Bsr': ${getEWAvgs(mxAd, mxBsr.t, n)}")
    println(s"Ad' + Bsr': ${getEWAvgs(mxAd.t, mxBsr.t, n)}")
    println(s"Ad'' + Bsr'': ${getEWAvgs(mxAd.t.t, mxBsr.t.t, n)}")
    println

    println(s"Asr + Bd: ${getEWAvgs(mxAsr, mxBd, n)}")
    println(s"Asr' + Bd: ${getEWAvgs(mxAsr.t, mxBd, n)}")
    println(s"Asr + Bd': ${getEWAvgs(mxAsr, mxBd.t, n)}")
    println(s"Asr' + Bd': ${getEWAvgs(mxAsr.t, mxBd.t, n)}")
    println(s"Asr'' + Bd'': ${getEWAvgs(mxAsr.t.t, mxBd.t.t, n)}")
    println

    println(s"Ad + Bsm: ${getEWAvgs(mxAd, mxBs, n)}")
    println(s"Ad' + Bsm: ${getEWAvgs(mxAd.t, mxBs, n)}")
    println(s"Ad + Bsm': ${getEWAvgs(mxAd, mxBs.t, n)}")
    println(s"Ad' + Bsm': ${getEWAvgs(mxAd.t, mxBs.t, n)}")
    println(s"Ad'' + Bsm'': ${getEWAvgs(mxAd.t.t, mxBs.t.t, n)}")
    println

    println(s"Asm + Bd: ${getEWAvgs(mxAs, mxBd, n)}")
    println(s"Asm' + Bd: ${getEWAvgs(mxAs.t, mxBd, n)}")
    println(s"Asm + Bd': ${getEWAvgs(mxAs, mxBd.t, n)}")
    println(s"Asm' + Bd': ${getEWAvgs(mxAs.t, mxBd.t, n)}")
    println(s"Asm'' + Bd'': ${getEWAvgs(mxAs.t.t, mxBd.t.t, n)}")
    println

    // Diagonal cases
    println(s"Ad + D: ${getEWAvgs(mxAd, mxD, n)}")
    println(s"Asr + D: ${getEWAvgs(mxAsr, mxD, n)}")
    println(s"Asm + D: ${getEWAvgs(mxAs, mxD, n)}")
    println(s"D + Ad: ${getEWAvgs(mxD, mxAd, n)}")
    println(s"D + Asr: ${getEWAvgs(mxD, mxAsr, n)}")
    println(s"D + Asm: ${getEWAvgs(mxD, mxAs, n)}")
    println

    println(s"Ad' + D: ${getEWAvgs(mxAd.t, mxD, n)}")
    println(s"Asr' + D: ${getEWAvgs(mxAsr.t, mxD, n)}")
    println(s"Asm' + D: ${getEWAvgs(mxAs.t, mxD, n)}")
    println(s"D + Ad': ${getEWAvgs(mxD, mxAd.t, n)}")
    println(s"D + Asr': ${getEWAvgs(mxD, mxAsr.t, n)}")
    println(s"D + Asm': ${getEWAvgs(mxD, mxAs.t, n)}")
    println

  }

  test("dense-view-debug") {
    val d = 500
    // Dense row-wise
    val mxAd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) + 1
    val mxBd = new DenseMatrix(d, d) := Matrices.gaussianView(d, d, 134) - 1

    mxAd(0 until mxAd.nrow, ::) %*% mxBd

  }
}
