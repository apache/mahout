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
import org.apache.mahout.math.drm.logical.{OpAewUnaryFunc, OpPar, OpMapBlock, OpRowRange}

/** Common Drm ops */
class DrmLikeOps[K](protected[drm] val drm: DrmLike[K]) {

  /**
   * Parallelism adjustments. <P/>
   *
   * Change only one of parameters from default value to choose new parallelism adjustment strategy.
   * <P/>
   *
   * E.g. use
   * <pre>
   *   drmA.par(auto = true)
   * </pre>
   * to use automatic parallelism adjustment.
   * <P/>
   *
   * Parallelism here in API is fairly abstract concept, and actual value interpretation is left for
   * a particular backend strategy. However, it is usually equivalent to number of map tasks or data
   * splits.
   * <P/>
   *
   * @param min If changed from default, ensures the product has at least that much parallelism.
   * @param exact if changed from default, ensures the pipeline product has exactly that much
   *              parallelism.
   * @param auto If changed from default, engine-specific automatic parallelism adjustment strategy
   *             is applied.
   */
  def par(min: Int = -1, exact: Int = -1, auto: Boolean = false) = {
    require(min > 0 || exact > 0 || auto, "Invalid argument")
    OpPar(drm, minSplits = min, exactSplits = exact)
  }

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
  def mapBlock[R: ClassTag](ncol: Int = -1, identicallyPartitioned: Boolean = true)
      (bmf: BlockMapFunc[K, R]): DrmLike[R] =
    new OpMapBlock[K, R](
      A = drm,
      bmf = bmf,
      _ncol = ncol,
      identicallyPartitioned = identicallyPartitioned
    )

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

    implicit val ktag = drm.keyClassTag

    val rowSrc: DrmLike[K] = if (rowRange != ::) {

      if (ClassTag.Int == ktag) {

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

  /**
    * Apply a function element-wise.
    *
    * @param f         element-wise function
    * @param evalZeros Do we have to process zero elements? true, false, auto: if auto, we will test
    *                  the supplied function for `f(0) != 0`, and depending on the result, will
    *                  decide if we want evaluation for zero elements. WARNING: the AUTO setting
    *                  may not always work correctly for functions that are meant to run in a specific
    *                  backend context, or non-deterministic functions, such as {-1,0,1} random
    *                  generators.
    * @return new DRM with the element-wise function applied.
    */
  def apply(f: Double ⇒ Double, evalZeros: AutoBooleanEnum.T = AutoBooleanEnum.AUTO) = {
    val ezeros = evalZeros match {
      case AutoBooleanEnum.TRUE ⇒ true
      case AutoBooleanEnum.FALSE ⇒ false
      case AutoBooleanEnum.AUTO ⇒ f(0) != 0
    }
    new OpAewUnaryFunc[K](drm, f, ezeros)
  }
}
