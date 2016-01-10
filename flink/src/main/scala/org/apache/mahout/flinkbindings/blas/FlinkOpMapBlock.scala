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

import scala.reflect.ClassTag

import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.drm.BlockMapFunc
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
 * Implementation is taken from Spark's MapBlock
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/MapBlock.scala
 */
object FlinkOpMapBlock {

  def apply[S, R: TypeInformation: ClassTag](src: FlinkDrm[S], ncol: Int, function: BlockMapFunc[S, R]): FlinkDrm[R] = {
    val res = src.asBlockified.ds.map {
      block =>
        val result = function(block)
        assert(result._2.nrow == block._2.nrow, "block mapping must return same number of rows.")
        assert(result._2.ncol == ncol, s"block map must return $ncol number of columns.")
        result
    }

    new BlockifiedFlinkDrm[R](res, ncol)
  }
}