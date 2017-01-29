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

import org.apache.mahout.math.{Vector => MahoutVector}

import collection._
import JavaConversions._
import org.apache.mahout.math._
import org.apache.mahout.math.drm.{drmBroadcast, _}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.{dvec, _}
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.reflect.ClassTag

class AsFactor extends PreprocessorModelFactory{

  def fit[K](input: DrmLike[K]) = {

    import org.apache.mahout.math.function.VectorFunction
    val factorMap = input.allreduceBlock(
      { case (keys, block) =>
        // someday we'll replace this with block.max: Vector
        // or better yet- block.distinct
        dense(block.aggregateColumns( new VectorFunction {
            def apply(f: Vector): Double = f.max
        }))
      })(0, ::)
    new AsFactorModel(factorMap.sum.toInt, factorMap)

  }

}

class AsFactorModel(k: Int, factorMap: MahoutVector) extends PreprocessorModel {

  def transform[K](input: DrmLike[K]): DrmLike[K] ={
    // fixed the 'fit' but transform still works in the old way.
    // not working!!
    //throw new Exception("asFactor doesn't work yet. Go home.")

    if (input.ncol != 1) {
      throw new Exception("This method is designed to work on singal columnar matrices")
    }

    implicit val ctx = input.context

    val bcastK = drmBroadcast(dvec(k))
    val bcastFactorMap = drmBroadcast(factorMap)

    implicit val ktag =  input.keyClassTag

    val res = input.mapBlock(k) {
      case (keys, block) => {
        val k: Int = bcastK.value.get(0).toInt
        val output = new SparseMatrix(block.nrow, bcastK.get(0).toInt)
        // This is how we take a vector of mapping to a map
        val fm = bcastFactorMap.all.toSeq.map(e => e.get -> e.index).toMap
        for (i <- 0 until output.nrow){
          output(i, ::) =  svec(fm.get(block.getQuick(i,0)).get -> 1.0 :: Nil, cardinality = bcastK.get(0).toInt)
        }
        (keys, output)
      }
    }
    res
  }

  override def invTransform[K](input: DrmLike[K]) = {
    // not yet implemented
    input

  }
}
