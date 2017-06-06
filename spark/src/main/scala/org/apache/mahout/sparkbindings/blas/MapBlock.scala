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

import org.apache.mahout.math.drm.logical.OpMapBlock
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.sparkbindings.drm.DrmRddInput

object MapBlock {

  def exec[S, R](src: DrmRddInput[S], operator:OpMapBlock[S,R]): DrmRddInput[R] = {

    // We can't use attributes directly in the closure in order to avoid putting the whole object
    // into closure.
    val bmf = operator.bmf
    val ncol = operator.ncol
    implicit val rtag = operator.keyClassTag
    src.asBlockified(operator.A.ncol).map(blockTuple => {
      val out = bmf(blockTuple)

      assert(out._2.nrow == blockTuple._2.nrow, "block mapping must return same number of rows.")
      assert(out._2.ncol == ncol, "block map must return %d number of columns.".format(ncol))

      out
    })
  }

}
