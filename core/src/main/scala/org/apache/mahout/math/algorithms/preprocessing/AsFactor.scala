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

package org.apache.mahout.math.algorithms.preprocessing

import collection._
import JavaConversions._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.{Vector => MahoutVector}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import MahoutCollections._

class AsFactor extends PreprocessorFitter {

  def fit[K](input: DrmLike[K],
             hyperparameters: (Symbol, Any)*): AsFactorModel = {

    import org.apache.mahout.math.function.VectorFunction
    val factorMap = input.allreduceBlock(
      { case (keys, block: Matrix) => block },
      { case (oldM: Matrix, newM: Matrix) =>
        // someday we'll replace this with block.max: Vector
        // or better yet- block.distinct

        dense((oldM rbind newM).aggregateColumns( new VectorFunction {
          def apply(f: Vector): Double = f.max
        }))
      })(0, ::)
    /*
    val A = drmParallelize(dense(
      (3, 2, 1),
      (0, 0, 0),
      (1, 1, 1))
      -> (4,2,2),  now 4,3,2
     */
    new AsFactorModel(factorMap.sum.toInt,
      dvec(factorMap.toArray.scanLeft(0.0)((l, r) => l + r ).take(factorMap.length))
    //  factorMap
    )
  }

}

class AsFactorModel(cardinality: Int, factorVec: MahoutVector) extends PreprocessorModel {

  val factorMap: MahoutVector = factorVec

  def transform[K](input: DrmLike[K]): DrmLike[K] ={

    implicit val ctx = input.context

    val bcastK = drmBroadcast(dvec(cardinality))
    val bcastFactorMap = drmBroadcast(factorMap)

    implicit val ktag =  input.keyClassTag

    val res = input.mapBlock(cardinality) {
      case (keys, block: Matrix) => {
        val cardinality: Int = bcastK.value.get(0).toInt
        val output = new SparseMatrix(block.nrow, cardinality)
        // This is how we take a vector of mapping to a map
        val fm = bcastFactorMap.value
        for (n <- 0 until output.nrow){
          var m = 0
          for (e <- block(n, ::).all() ){
            output(n, fm.get(m).toInt + e.get().toInt ) = 1.0
            m += 1
          }
        }
        (keys, output)
      }
    }
    res
  }

  override def invTransform[K](input: DrmLike[K]): DrmLike[K] = {
    implicit val ctx = input.context

    val bcastK = drmBroadcast(dvec(cardinality))
    val bcastFactorMap = drmBroadcast(factorMap)

    implicit val ktag =  input.keyClassTag

    val res = input.mapBlock(cardinality) {
      case (keys, block: Matrix) => {
        val k: Int = bcastK.value.get(0).toInt
        val output = new DenseMatrix(block.nrow, bcastK.value.length)
        // This is how we take a vector of mapping to a map
        val fm = bcastFactorMap.all.toSeq.map(e => e.get -> e.index).toMap

        import MahoutCollections._
        val indexArray = Array(1.0) ++ bcastFactorMap.value.toArray.map(i => i.toInt)
        for (n <- 0 until output.nrow){
          val v = new DenseVector(bcastFactorMap.value.length)
          var m = 0
          for (e <- block(n, ::).asInstanceOf[RandomAccessSparseVector].iterateNonZero() ){
            v.setQuick(m, e.index - m)
            m += 1
          }
          output(n, ::) = v
        }
        (keys, output)
      }
    }
    res
  }

}
