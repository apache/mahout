package org.apache.mahout.math.drm.decompositions

import scala.reflect.ClassTag
import org.apache.mahout.math._
import drm._
import scalabindings._
import RLikeDrmOps._
import RLikeOps._
import scala.util.Random
import org.apache.log4j.Logger

/** Simple ALS factorization algotithm. To solve, use train() method. */
object ALS {

  private val log = Logger.getLogger(ALS.getClass)

  /**
   * ALS training result. <P>
   *
   * <code>drmU %*% drmV.t</code> is supposed to approximate the input.
   *
   * @param drmU U matrix
   * @param drmV V matrix
   * @param iterationsRMSE RMSE values afeter each of iteration performed
   */
  class Result[K: ClassTag](val drmU: DrmLike[K], val drmV: DrmLike[Int], val iterationsRMSE: Iterable[Double]) {
    def toTuple = (drmU, drmV, iterationsRMSE)
  }

  /**
   * Run ALS.
   * <P>
   *
   * Example:
   *
   * <pre>
   *   val (u,v,errors) = train(input, k).toTuple
   * </pre>
   *
   * ALS runs until (rmse[i-1]-rmse[i])/rmse[i-1] < convergenceThreshold, or i==maxIterations,
   * whichever earlier.
   * <P>
   *
   * @param drmInput The input matrix
   * @param k required rank of decomposition (number of cols in U and V results)
   * @param convergenceThreshold stop sooner if (rmse[i-1] - rmse[i])/rmse[i - 1] is less than this
   *                             value. If <=0 then we won't compute RMSE and use convergence test.
   * @param maxIterations maximum iterations to run regardless of convergence
   * @tparam K row key type of the input (100 is probably more than enough)
   * @return { @link org.apache.mahout.math.drm.decompositions.ALS.Result}
   */
  def train[K: ClassTag](
      drmInput: DrmLike[K],
      k: Int = 50,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.10
      ): Result[K] = {

    assert(convergenceThreshold < 1.0, "convergenceThreshold")

    val drmA = drmInput
    val drmAt = drmInput.t

    // Initialize U and V so that they are identically distributed to A or A'
    var drmU = drmA.mapBlock() {
      case (keys, block) =>
        val uBlock = Matrices.symmetricUniformView(block.nrow, k, Random.nextInt()) * 0.01
        keys -> uBlock
    }

    var drmV:DrmLike[Int] = null
    var rmseIterations:List[Double] = Nil

    // ALS iterator
    var stop = false
    var i = 0
    while (! stop && i < maxIterations) {
      drmV = drmAt %*% drmU %*% solve(drmU.t %*% drmU)
      drmU = drmA %*% drmV %*% solve(drmV.t %*% drmV)
      i += 1

      // Check if we are requested to do a convergence test.
      if (convergenceThreshold>0) {
        // Compute rmse and test convergence
        val rmse = (drmA - drmU %*% drmV.t).norm

        if (i > 0 ) {
          val rmsePrev = rmseIterations.last
          val convergence = (rmsePrev - rmse) / rmsePrev

          if (convergence <0 ) {
            log.warn("Rmse increase of %f. Should not happen.".format(convergence))
          } else if ( convergence < convergenceThreshold ) {
            stop=true
          }
        }
        rmseIterations :+= rmse
      }
    }

    new Result(drmU,drmV,rmseIterations)
  }


}
