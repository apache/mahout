package org.apache.mahout.math.decompositions

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import scala.reflect.ClassTag
import scala.util.Random
import org.apache.spark.graphx._

/**
 *
 * @author dmitriy
 */
object ALSImplicit {

  def alsImplicit(
      drmInput: DrmLike[Int],
      k: Int = 50,
      lambda: Double = 0.01,
      maxIterations: Int = 10,
      convergenceTreshold: Double = 0.10
      ) = {
    val drmA = drmInput
    val drmAt = drmInput.t

    // cbind(U,A):
    var drmUA = drmA.mapBlock(ncol = k + drmA.ncol) {
      case (keys, block) =>
        val uaBlock = new SparseRowMatrix(block.nrow, block.ncol + k)
        uaBlock(::, 0 until k) := Matrices.symmetricUniformView(uaBlock.nrow, k, Random.nextInt()) * 0.01
        uaBlock(::, k until uaBlock.ncol) := block
        keys -> uaBlock
    }

    // cbind(V,A'):
    val drmVAt = drmAt.mapBlock(ncol = k + drmAt.ncol) {
      case (keys, block) =>
        val vatBlock = new SparseRowMatrix(block.nrow, block.ncol + k)
        vatBlock(::, k until vatBlock.ncol) := block
        keys -> vatBlock
    }
  }

  private def buildGraph[K:ClassTag](drmInput:DrmLike[K]) = {
    // Build preference 1 edges (indicator matrix where confidence > 0)
//    val edges = drmInput.m

    // Graph()


  }

}
