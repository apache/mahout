/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */

package org.apache.mahout.math.algorithms.neuralnet

import org.apache.mahout.math.{DenseMatrix, DenseVector, Matrix, Vector}
import org.apache.mahout.math.scalabindings.RLikeOps._

object Converters {

  def flattenMatrixToVec(m: Matrix): Vector = {
    val v = new DenseVector(m.nrow * m.ncol)
    for (i <- 0 until m.nrow){
      for (j <- 0 until m.ncol){
        v.setQuick((i * m.ncol) + j, m.getQuick(i,j) )
      }
    }
    v
  }

  def recomposeMatrixFromVec(v: Vector, rows: Int, columns: Int) = {
    val m2 = new DenseMatrix(rows, columns)
    for (i <- 0 until rows){
      m2.assignRow(i, v.viewPart(i * columns, columns) )
    }
    m2
  }

  def flattenMatrixArrayToVector(mA: Array[_ <: Matrix]): Vector = {
    val sizes = mA.map(m => m.nrow * m.ncol)
    val offsets = sizes.scanLeft(0)(_ + _)
    val v = new DenseVector(sizes.reduce(_ + _))
    for (ai <- mA.indices){
      val vPart = flattenMatrixToVec(mA(ai))
      for (vi <- 0 until vPart.length){
        v.setQuick(offsets(ai) + vi, vPart(vi))
      }
    }
    v
  }

  def recomposeMatrixArrayFromVec(v: Vector, sizeArray: Array[(Int, Int)]): Array[Matrix] = {
    val sizes = sizeArray.map( s => s._1 * s._2)
    val offsets = sizes.scanLeft(0)(_ + _)

    val mA = new Array[Matrix](sizeArray.length)
    for (i <- sizeArray.indices){
      val rows = sizeArray(i)._1
      val cols = sizeArray(i)._2
      mA(i) = recomposeMatrixFromVec(v.viewPart(offsets(i), sizes(i)), rows, cols)
    }

    mA
  }

  def flattenArrayOfMatrixArraysToVector(aMA: Array[Array[_ <: Matrix]]): Vector = {
    val sizes = aMA.map(aM => aM.map(m => m.nrow * m.ncol).sum)
    val offsets = sizes.scanLeft(0)(_ + _)
    val v = new DenseVector(sizes.sum)
    for (ai <- aMA.indices){
      val vPart = flattenMatrixArrayToVector(aMA(ai))
      for (vi <- 0 until vPart.length){
        v.setQuick(offsets(ai) + vi, vPart(vi))
      }
    }
    v
  }

  def recomposeArrayOfMatrixArraysFromVec(v: Vector, sizeArray: Array[Array[(Int, Int)]]): Array[Array[Matrix]] = {
    val sizes = sizeArray.map(aM => aM.map(m => m._1 * m._2).sum)
    val offsets = sizes.scanLeft(0)(_ + _)
    val cols = sizeArray.map(ma => ma.length).max
    val aMA = Array.ofDim[Matrix](sizeArray.length, cols)
    for (i <- 0 until sizeArray.length){
      for (j <- 0 until sizeArray(i).length){
        aMA(i) = recomposeMatrixArrayFromVec( v.viewPart(offsets(i), sizes(i)), sizeArray(i))
      }

    }
    aMA
  }
}
