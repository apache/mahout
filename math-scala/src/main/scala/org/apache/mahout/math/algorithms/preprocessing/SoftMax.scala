

package org.apache.mahout.math.algorithms.preprocessing

import collection._
import JavaConversions._

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.function.{Functions => F}
import org.apache.mahout.math.scalabindings.RLikeOps._

class SoftMaxPostprocessor extends PreprocessorModel {

  def transform[K](input: DrmLike[K]): DrmLike[K] = {

    implicit val ktag =  input.keyClassTag
    input.mapBlock(input.ncol) {
      case (keys, block: Matrix) =>
        val copy: Matrix = block.cloned
        copy.foreach(v => v := v.assign(F.EXP) / v.assign(F.EXP).zSum())
        (keys, copy)
    }
  }

  override def invTransform[K](input: DrmLike[K]) = throw new NotImplementedError("softmax inverse not implemented")

}
