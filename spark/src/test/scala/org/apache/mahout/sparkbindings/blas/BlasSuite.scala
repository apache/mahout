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

package org.apache.mahout.sparkbindings.blas

import java.io.ByteArrayOutputStream

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.Output
import com.twitter.chill.AllScalaRegistrar
import org.apache.log4j.Level
import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.{OpABt, OpAewB, OpAt, OpAtA}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.FunSuite

/** Collection of physical blas operator tests. */
class BlasSuite extends FunSuite with DistributedSparkSuite {

  private final implicit val mahoutLog = getLog(classOf[RLikeDrmOpsSuite])

  test("ABt") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val inCoreB = dense((3, 4, 5), (5, 6, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 3)
    val drmB = drmParallelize(m = inCoreB, numPartitions = 2)

    val op = OpABt(drmA, drmB)

    val drm = new CheckpointedDrmSpark(ABt.abt(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    printf("AB' num partitions = %d.\n", drm.rdd.partitions.size)

    val inCoreMControl = inCoreA %*% inCoreB.t
    val inCoreM = drm.collect

    assert((inCoreM - inCoreMControl).norm < 1E-5)

    println(inCoreM)
  }

  test("A * B Hadamard") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = OpAewB(drmA, drmB, "*")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA * inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A + B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = OpAewB(drmA, drmB, "+")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA + inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A - B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = OpAewB(drmA, drmB, "-")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA - inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A / B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 0), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (10, 20, 30), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = OpAewB(drmA, drmB, "/")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA / inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("AtA slim") {

    val inCoreA = dense((1, 2), (2, 3))
    val drmA = drmParallelize(inCoreA)

    val operator = new OpAtA[Int](A = drmA)
    val inCoreAtA = AtA.at_a_slim(operator = operator, srcRdd = drmA.rdd)
    println(inCoreAtA)

    val expectedAtA = inCoreA.t %*% inCoreA
    println(expectedAtA)

    assert(expectedAtA === inCoreAtA)

  }

  test("At") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)

    val op = OpAt(drmA)
    val drmAt = new CheckpointedDrmSpark(rddInput = At.at(op, srcA = drmA), _nrow = op.nrow, _ncol = op.ncol)
    val inCoreAt = drmAt.collect
    val inCoreControlAt = inCoreA.t

    println(inCoreAt)
    assert((inCoreAt - inCoreControlAt).norm < 1E-5)

  }

  test("verbosity") {
    def testreg(o: Any*): Unit = {
      val s = new String(kryoSet(o: _*))
      s.contains("org.apache.mahout") shouldBe false
    }

    def kryoSet[T](obj: T*) = {

      val kryo = new Kryo()
      new AllScalaRegistrar()(kryo)

      MahoutKryoRegistrator.registerClasses(kryo)

      val baos = new ByteArrayOutputStream()
      val output = new Output(baos)
      obj.foreach(kryo.writeClassAndObject(output, _))
      output.close

      baos.toByteArray
    }

    mahoutLog.setLevel(Level.TRACE)

    val mxA = dense((1, 2), (3, 4))
    val mxB = new SparseRowMatrix(4,5)
    val mxC = new SparseMatrix(4,5)
    val mxD = diagv(dvec(1, 2, 3, 5))
    val mxE = mxA (0 to 0, 0 to 0)
    val mxF = mxA.t


    testreg(
      mxD, mxD(0, ::), mxD(::, 0), mxD.diagv,
      mxA, mxA(0, ::), mxA(::, 0), mxA.diagv,
      mxB, mxB(0, ::), mxB(::, 0), mxB.diagv,
      mxC, mxC(0, ::), mxC(::, 0), mxC.diagv,
      mxE, mxE(0, ::), mxE(::, 0), mxE.diagv,
      mxF, mxF(0, ::), mxF(::, 0), mxF.diagv,
      mxA(0,::)(0 to 0), mxE(0,::)(0 to 0),
      new DenseVector(6), new DenseVector(6) (0 to 0),
      new RandomAccessSparseVector(6), new RandomAccessSparseVector(6)(0 to 0),
      new SequentialAccessSparseVector(6), new SequentialAccessSparseVector(6)(0 to 0)

    )

  }

}
