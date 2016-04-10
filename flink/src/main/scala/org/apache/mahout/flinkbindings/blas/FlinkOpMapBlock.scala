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
package org.apache.mahout.flinkbindings.blas

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.drm.logical.OpMapBlock
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
 * Implementation of Flink's MapBlock
 */
object FlinkOpMapBlock {

  def apply[S, R: TypeInformation](src: FlinkDrm[S], ncol: Int, operator: OpMapBlock[S,R]): FlinkDrm[R] = {

    implicit val rtag = operator.keyClassTag
    val bmf = operator.bmf
    val ncol = operator.ncol
    val res = src.asBlockified.ds.map {
      block =>
        val result = bmf(block)
        assert(result._2.nrow == block._2.nrow, "block mapping must return same number of rows.")
        assert(result._2.ncol == ncol, s"block map must return $ncol number of columns.")
        result
    }

    new BlockifiedFlinkDrm[R](res, ncol)
  }
}