package org.apache.mahout.math.decompositions

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import scala.util.Random
import scala.collection.JavaConversions._
import org.apache.spark.SparkContext._

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
        val uaBlock = block.like(block.nrow, block.ncol + k)
        uaBlock(::, 0 until k) := Matrices.symmetricUniformView(uaBlock.nrow, k, Random.nextInt()) * 0.01
        uaBlock(::, k until uaBlock.ncol) := block
        keys -> uaBlock
    }

    // cbind(V,A'):
    val drmVAt = drmAt.mapBlock(ncol = k + drmAt.ncol) {
      case (keys, block) =>
        val vatBlock = block.like(block.nrow, block.ncol + k)
        vatBlock(::, k until vatBlock.ncol) := block
        keys -> vatBlock
    }
    
    var i = 0 
    var stop = false
    while (i < maxIter && ! stop) {
      // update VAt, TODO to be ctd.
      drmVAt.rdd.cogroup(other=generateMessages(drmUA,k)).map()

      i+= 1 
    }
  }

  private def generateMessages(drmUA:DrmLike[Int], k:Int) = {

    val n = drmUA.ncol
    // Now we delve into Spark-specific processing.
    drmUA.rdd.flatMap{
      case (rowKey, row) =>
      val uvec = new DenseVector(k) := row(0 until k)
      val cvec = row(k until n)
      cvec.nonZeroes().map(_.index()-> uvec)
    }
  }

}
