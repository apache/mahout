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

package org.apache.mahout.math.drm

import scala.reflect.ClassTag
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm.logical.{OpMapBlock, OpRowRange}

/** Common Drm ops */
class DrmLikeOps[K : ClassTag](protected[drm] val drm: DrmLike[K]) {

  /**
   * Map matrix block-wise vertically. Blocks of the new matrix can be modified original block
   * matrices; or they could be completely new matrices with new keyset. In the latter case, output
   * matrix width must be specified with <code>ncol</code> parameter.<P>
   *
   * New block heights must be of the same height as the original geometry.<P>
   *
   * @param ncol new matrix' width (only needed if width changes).
   * @param bmf
   * @tparam R
   * @return
   */
  def mapBlock[R : ClassTag](ncol: Int = -1)
      (bmf: BlockMapFunc[K, R]): DrmLike[R] =
    new OpMapBlock[K, R](A = drm, bmf = bmf, _ncol = ncol)


  /**
   * Slicing the DRM. Should eventually work just like in-core drm (e.g. A(0 until 5, 5 until 15)).<P>
   *
   * The all-range is denoted by '::', e.g.: A(::, 0 until 5).<P>
   *
   * Row range is currently unsupported except for the all-range. When it will be fully supported,
   * the input must be Int-keyed, i.e. of DrmLike[Int] type for non-all-range specifications.
   *
   * @param rowRange Row range. This must be '::' (all-range) unless matrix rows are keyed by Int key.
   * @param colRange col range. Must be a sub-range of <code>0 until ncol</code>. '::' denotes all-range.
   */
  def apply(rowRange: Range, colRange: Range): DrmLike[K] = {

    import RLikeDrmOps._
    import RLikeOps._

    val rowSrc: DrmLike[K] = if (rowRange != ::) {

      if (implicitly[ClassTag[Int]] == implicitly[ClassTag[K]]) {

        assert(rowRange.head >= 0 && rowRange.last < drm.nrow, "rows range out of range")
        val intKeyed = drm.asInstanceOf[DrmLike[Int]]

        new OpRowRange(A = intKeyed, rowRange = rowRange).asInstanceOf[DrmLike[K]]

      } else throw new IllegalArgumentException("non-all row range is only supported for Int-keyed DRMs.")

    } else drm

    if (colRange != ::) {

      assert(colRange.head >= 0 && colRange.last < drm.ncol, "col range out of range")

      // Use mapBlock operator to do in-core subranging.
      rowSrc.mapBlock(ncol = colRange.length)({
        case (keys, block) => keys -> block(::, colRange)
      })

    } else rowSrc
  }
}
