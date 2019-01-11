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

package org.apache.mahout.sparkbindings.io

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.twitter.chill.AllScalaRegistrar
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

import org.apache.mahout.common.RandomUtils
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

import scala.util.Random

class IOSuite extends FunSuite with MahoutSuite {

  import IOSuite._

  test("Dense vector kryo") {

    val rnd = RandomUtils.getRandom
    val vec = new DenseVector(165) := { _ => rnd.nextDouble()}

    val ret = kryoClone(vec, vec, vec)
    val vec2 = ret(2)

    println(s"vec=$vec\nvc2=$vec2")

    vec2 === vec shouldBe true
    vec2.isInstanceOf[DenseVector] shouldBe true
  }

  test("Random sparse vector kryo") {

    val rnd = RandomUtils.getRandom
    val vec = new RandomAccessSparseVector(165) := { _ => if (rnd.nextDouble() < 0.3) rnd.nextDouble() else 0}
    val vec1 = new RandomAccessSparseVector(165)
    vec1(2) = 2
    vec1(3) = 4
    vec1(3) = 0
    vec1(10) = 30

    val ret = kryoClone(vec, vec1, vec)
    val (vec2, vec3) = (ret(2), ret(1))

    println(s"vec=$vec\nvc2=$vec2")

    vec2 === vec shouldBe true
    vec1 === vec3 shouldBe true
    vec2.isInstanceOf[RandomAccessSparseVector] shouldBe true

  }

  test("100% sparse vectors") {

    val vec1 = new SequentialAccessSparseVector(10)
    val vec2 = new RandomAccessSparseVector(6)
    val ret = kryoClone(vec1, vec2, vec1, vec2)
    val vec3 = ret(2)
    val vec4 = ret(3)

    vec1 === vec3 shouldBe true
    vec2 === vec4 shouldBe true
  }

  test("Sequential sparse vector kryo") {

    val rnd = RandomUtils.getRandom
    val vec = new SequentialAccessSparseVector(165) := { _ => if (rnd.nextDouble() < 0.3) rnd.nextDouble() else 0}

    val vec1 = new SequentialAccessSparseVector(165)
    vec1(2) = 0
    vec1(3) = 3
    vec1(4) = 2
    vec1(3) = 0

    val ret = kryoClone(vec, vec1, vec)
    val (vec2, vec3) = (ret(2), ret(1))

    println(s"vec=$vec\nvc2=$vec2")

    vec2 === vec shouldBe true
    vec1 === vec3 shouldBe true
    vec2.isInstanceOf[SequentialAccessSparseVector] shouldBe true
  }

  test("kryo matrix tests") {
    val rnd = new Random()

    val mxA = new DenseMatrix(140, 150) := { _ => rnd.nextDouble()}

    val mxB = new SparseRowMatrix(140, 150) := { _ => if (rnd.nextDouble() < .3) rnd.nextDouble() else 0.0}

    val mxC = new SparseMatrix(140, 150)
    for (i <- 0 until mxC.nrow) if (rnd.nextDouble() < .3)
      mxC(i, ::) := { _ => if (rnd.nextDouble() < .3) rnd.nextDouble() else 0.0}

    val cnsl = mxC.numSlices()
    println(s"Number of slices in mxC: $cnsl")

    val ret = kryoClone(mxA, mxA.t, mxB, mxB.t, mxC, mxC.t, mxA)

    val (mxAA, mxAAt, mxBB, mxBBt, mxCC, mxCCt, mxAAA) = (ret.head, ret(1), ret(2), ret(3), ret(4), ret(5), ret(6))

    // ret.size shouldBe 7

    mxA === mxAA shouldBe true
    mxA === mxAAA shouldBe true
    mxA === mxAAt.t shouldBe true
    mxAA.isInstanceOf[DenseMatrix] shouldBe true
    mxAAt.isInstanceOf[DenseMatrix] shouldBe false


    mxB === mxBB shouldBe true
    mxB === mxBBt.t shouldBe true
    mxBB.isInstanceOf[SparseRowMatrix] shouldBe true
    mxBBt.isInstanceOf[SparseRowMatrix] shouldBe false
    mxBB(0,::).isDense shouldBe false


    // Assert no persistence operation increased slice sparsity
    mxC.numSlices() shouldBe cnsl

    // Assert deserialized product did not experience any empty slice inflation
    mxCC.numSlices() shouldBe cnsl
    mxCCt.t.numSlices() shouldBe cnsl

    // Incidentally, but not very significantly, iterating thru all rows that happens in equivalence
    // operator, inserts empty rows into SparseMatrix so these asserts should not be before numSlices
    // asserts.
    mxC === mxCC shouldBe true
    mxC === mxCCt.t shouldBe true
    mxCCt.t.isInstanceOf[SparseMatrix] shouldBe true

    // Column-wise sparse matrix are deprecated and should be explicitly rejected by serializer.
    an[IllegalArgumentException] should be thrownBy {
      val mxDeprecated = new SparseColumnMatrix(14, 15)
      kryoClone(mxDeprecated)
    }

  }

  test("diag matrix") {

    val mxD = diagv(dvec(1, 2, 3, 5))
    val mxDD = kryoClone(mxD).head
    mxD === mxDD shouldBe true
    mxDD.isInstanceOf[DiagonalMatrix] shouldBe true

  }
}

object IOSuite {

  def kryoClone[T](obj: T*): Seq[T] = {

    val kryo = new Kryo()
    new AllScalaRegistrar()(kryo)

    MahoutKryoRegistrator.registerClasses(kryo)

    val baos = new ByteArrayOutputStream()
    val output = new Output(baos)
    obj.foreach(kryo.writeClassAndObject(output, _))
    output.close

    val input = new Input(new ByteArrayInputStream(baos.toByteArray))

    def outStream: Stream[T] =
      if (input.eof) Stream.empty
      else kryo.readClassAndObject(input).asInstanceOf[T] #:: outStream

    outStream
  }
}
