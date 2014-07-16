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

import org.apache.log4j.Logger
import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm.logical.OpRbind
import org.apache.spark.SparkContext._

/** Physical Rbind */
object RbindAB {

  private val log = Logger.getLogger(RbindAB.getClass)

  def rbindAB_int[K:ClassTag](op: OpRbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    val a = srcA.asInstanceOf[DrmRddInput[Int]].toDrmRdd()
    val b = srcB.asInstanceOf[DrmRddInput[Int]].toDrmRdd()
    val n1 = op.A.nrow.asInstanceOf[Int]

    val rdd = a union (b map({ case (key, vec) => ((key + n1), vec) }))

    new DrmRddInput(rowWiseSrc = Some(op.ncol -> rdd)).asInstanceOf[DrmRddInput[K]]
  }

  def rbindAB_long[K:ClassTag](op: OpRbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    val a = srcA.asInstanceOf[DrmRddInput[Long]].toDrmRdd()
    val b = srcB.asInstanceOf[DrmRddInput[Long]].toDrmRdd()
    val n1 = op.A.nrow

    val rdd = a union (b map({ case (key, vec) => ((key + n1), vec) }))

    new DrmRddInput(rowWiseSrc = Some(op.ncol -> rdd)).asInstanceOf[DrmRddInput[K]]
  }

  def rbindAB_string[K:ClassTag](op: OpRbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    val a = srcA.asInstanceOf[DrmRddInput[String]].toDrmRdd()
    val b = srcB.asInstanceOf[DrmRddInput[String]].toDrmRdd()
    val n1 = op.A.nrow

    val rdd = a union b

    new DrmRddInput(rowWiseSrc = Some(op.ncol -> rdd)).asInstanceOf[DrmRddInput[K]]
  }

  def rbindAB[K: ClassTag](op: OpRbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {
    if (implicitly[ClassTag[K]] == implicitly[ClassTag[Int]])
      rbindAB_int(op, srcA, srcB)
    else if (implicitly[ClassTag[K]] == implicitly[ClassTag[Long]])
      rbindAB_long(op, srcA, srcB)
    else if (implicitly[ClassTag[K]] == implicitly[ClassTag[String]])
      rbindAB_string(op, srcA, srcB)
    else
      throw new IllegalArgumentException("Unsupported Key type.")
  }
}
