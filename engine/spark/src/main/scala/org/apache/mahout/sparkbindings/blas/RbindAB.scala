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
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.drm.logical.OpRbind

/** Physical `rbind` */
object RbindAB {

  private val log = Logger.getLogger(RbindAB.getClass)

  def rbindAB[K](op: OpRbind[K], srcA: DrmRddInput[K], srcB: DrmRddInput[K]): DrmRddInput[K] = {

    implicit val ktag = op.keyClassTag

    // If any of the inputs is blockified, use blockified inputs
    if (srcA.isBlockified || srcB.isBlockified) {
      val a = srcA.asBlockified(op.A.ncol)
      val b = srcB.asBlockified(op.B.ncol)

      // Union seems to be fine, it is indeed just do partition-level unionization, no shuffles
      a ++ b

    } else {

      // Otherwise, use row-wise inputs -- no reason to blockify here.
      val a = srcA.asRowWise()
      val b = srcB.asRowWise()

      a ++ b
    }
  }
}
