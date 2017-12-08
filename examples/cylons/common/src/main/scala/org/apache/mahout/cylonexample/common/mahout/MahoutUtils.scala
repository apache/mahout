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

package org.apache.mahout.cylonexample.common.mahout

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.ByteBuffer

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings.MahoutCollections._

import scala.collection.JavaConversions._

object MahoutUtils {

  def matrixWriter(m: Matrix, path: String): Unit = {
    val oos = new ObjectOutputStream(new FileOutputStream(path))
    oos.writeObject(m.toArray.map(v => v.toArray).toList)
    oos.close
  }

  def matrixReader(path: String): Matrix ={
    val ois = new ObjectInputStream(new FileInputStream(path))
    // for row in matrix
    val la = ois.readObject.asInstanceOf[List[Array[Double]]]
    ois.close
    val m = listArrayToMatrix(la)
    m
  }

  def vectorReader(path: String): Vector ={
    val m = matrixReader(path)
    m(0, ::)
  }

  def listArrayToMatrix(la: List[Array[Double]]): Matrix = {
    dense(la.map(m => dvec(m)):_*)
  }

  def decomposeImgVecWithEigenfaces(v: Vector, m: Matrix): Vector = {

    val XtX = m.t %*% m
    val Xty = m.t %*% v
    solve(XtX, Xty).viewPart(3, m.numCols()-3)  // The first 3 eigenfaces often only capture 3 dimensional light, which we want to ignore

  }

  def vector2byteArray(v: Vector): Array[Byte] = {
    val bb: ByteBuffer = ByteBuffer.allocate(v.size() * 8)
    for (d <- v.toArray){
      bb.putDouble(d)
    }
    bb.array()
  }

  def byteArray2vector(ba: Array[Byte]): Vector = {
    val bb: ByteBuffer = ByteBuffer.wrap(ba)
    val output: Array[Double] = new Array[Double](ba.length / 8)
    for (i <- output.indices) {
      output(i) = bb.getDouble()
    }
    dvec(output)
  }
}
