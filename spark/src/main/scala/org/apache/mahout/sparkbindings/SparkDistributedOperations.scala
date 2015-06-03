/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.sparkbindings

import org.apache.mahout.math.drm.{DrmLike, BlockifiedDrmTuple, CheckpointedDrm, DistributedOperations}
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.Vector

import scala.reflect.ClassTag

object SparkDistributedOperations extends DistributedOperations{
  private def toDrmRddInput[K: ClassTag] (drm: DrmLike[K]): DrmRddInput[K] = {
    val cp = drm match {
      case a: CheckpointedDrm[K] => a
      case _ => drm.checkpoint()
    }
    new DrmRddInput[K](rowWiseSrc = Some((cp.ncol, cp.rdd)))
  }



  override def aggregateBlocks[U: ClassTag, K: ClassTag](drm: DrmLike[K])
                                                        (zeroValue: U, seqOp: (U, BlockifiedDrmTuple[K]) => U,
                                                         combOp: (U, U) => U): U = {
    val out = toDrmRddInput(drm).toBlockifiedDrmRdd().aggregate(zeroValue)(seqOp, combOp)
    out

  }

  override def aggregateRows[U: ClassTag, K: ClassTag](drm: DrmLike[K])
                                                      (zeroValue: U, seqOp: (U, (K, Vector)) => U,
                                                       combOp: (U, U) => U): U = {
    val out = toDrmRddInput(drm).toDrmRdd().aggregate(zeroValue)(seqOp, combOp)
    out
  }

}
