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

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm._
import org.apache.spark.rdd.RDD
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.spark.SparkContext._
import org.apache.mahout.sparkbindings.drm.plan.{OpAtB}
import org.apache.log4j.Logger

object AtB {

  private val log = Logger.getLogger(AtB.getClass)

  /**
   * The logic for computing A'B is pretty much map-side generation of partial outer product blocks
   * over co-grouped rows of A and B. If A and B are identically partitioned, we can just directly
   * zip all the rows. Otherwise, we need to inner-join them first.
   */
  def atb_nograph[A: ClassTag](
      operator: OpAtB[A],
      srcA: DrmRddInput[A],
      srcB: DrmRddInput[A],
      zippable:Boolean = false
      ): DrmRddInput[Int] = {

    val rddA = srcA.toDrmRdd()
    val zipped = if ( zippable ) {

      log.debug("A and B for A'B are identically distributed, performing row-wise zip.")

      rddA.zip(other = srcB.toDrmRdd())

    } else {

      log.debug("A and B for A'B are not identically partitioned, performing inner join.")

      rddA.join(other=srcB.toDrmRdd()).map({
        case (key,(v1,v2) ) => (key -> v1) -> (key -> v2)
      })
    }

    val blockHeight = safeToNonNegInt(
      (operator.B.ncol.toDouble/rddA.partitions.size).ceil.round max 1L
    )

    computeAtBZipped(
      zipped,
      nrow = operator.nrow,
      ancol = operator.A.ncol,
      bncol = operator.B.ncol,
      blockHeight = blockHeight
    )
  }


//  private[sparkbindings] def atb_nograph()

  /** Given already zipped, joined rdd of rows of A' and B, compute their product A'B */
  private[sparkbindings] def computeAtBZipped[A: ClassTag](zipped:RDD[(DrmTuple[A], DrmTuple[A])],
      nrow:Long, ancol:Int, bncol:Int, blockHeight: Int) = {

    // Since Q and A are partitioned same way,we can just zip their rows and proceed from there by
    // forming outer products. Our optimizer lacks this primitive, so we will implement it using RDDs
    // directly. We try to compile B' = A'Q now by collecting outer products of rows of A and Q. At
    // this point we need to split n-range  of B' into sutiable number of partitions.

    val btNumParts = safeToNonNegInt((nrow - 1) / blockHeight + 1)

    val rddBt = zipped

        // Produce outer product blocks
        .flatMap {
      case ((aKey, aRow), (qKey, qRow)) =>
        for (blockKey <- Stream.range(0, btNumParts)) yield {
          val blockStart = blockKey * blockHeight
          val blockEnd = ancol min (blockStart + blockHeight)

          // Create block by cross product of proper slice of aRow and qRow
          blockKey -> (aRow(blockStart until blockEnd) cross qRow)
        }
    }
        // Combine blocks by just summing them up
        .reduceByKey {
      case (block1, block2) => block1 += block2
    }

        // Throw away block key, generate row keys instead.
        .map {
      case (blockKey, block) =>
        val blockStart = blockKey * blockHeight
        val rowKeys = Array.tabulate(block.nrow)(blockStart + _)
        rowKeys -> block
    }

    new DrmRddInput[Int](blockifiedSrc = Some(rddBt))
  }

}
