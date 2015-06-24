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

import org.apache.flink.api.common.functions.FilterFunction
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.logical.OpRowRange

/**
 * Implementation is taken from Spark's OpRowRange
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/Slicing.scala
 */
object FlinkOpRowRange {

  def slice(op: OpRowRange, A: FlinkDrm[Int]): FlinkDrm[Int] = {
    val rowRange = op.rowRange
    val firstIdx = rowRange.head

    val filtered = A.deblockify.ds.filter(new FilterFunction[(Int, Vector)] {
      def filter(tuple: (Int, Vector)): Boolean = tuple match {
        case (idx, vec) => rowRange.contains(idx)
      }
    })

    val res = filtered.map(new MapFunction[(Int, Vector), (Int, Vector)] {
      def map(tuple: (Int, Vector)): (Int, Vector) = tuple match {
        case (idx, vec) => (idx - firstIdx, vec)
      }
    })

    new RowsFlinkDrm(res, op.ncol)
  }

}