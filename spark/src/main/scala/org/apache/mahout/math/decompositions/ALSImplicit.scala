package org.apache.mahout.math.decompositions

import math._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import scala.collection.JavaConversions._
import org.apache.spark.SparkContext._
import org.apache.mahout.common.RandomUtils

object ALSImplicit {

  /**
   * See MAHOUT-1365 for details.
   *
   * Implicit feedback has two inputs: the preference matrix and confidence matrix. Preference on
   * user/item interaciton is always 1 (prefer) or 0 (do not prefer). Confidence encodes informal
   * weighing on how confident we are about preference. Things we don't have any observations for
   * are usually encoded with baseline confidence (c0 parameter).Things we do get observations for
   * have higher confidence. Thus, it is assumed that all confidence cells are greater or equal c0.
   * <P/>
   *
   * Next, for the purposes of compactness of input, we sparsify and combine both inputs in the
   * following way.
   * <P/>
   *
   * First, we assume that we don't have observations for every combination of user/item pair, so
   * (C-c0) matrix is sparse.
   * <P/>
   *
   * Second, we can use sign to encode preferences, i.e.
   * <pre>
   *     C*(i,j) = (C(i,j)-c0) if P(i,j)==1;
   *          and
   *     C*(i,j) = -(C(i,j)-c0) if P(i,j)=0.
   * </pre>
   *
   * Note in that we assume all entries with baseline confidence having P = 0 (no preference).
   * <P/>
   *
   * In reality this input vectorizes without need to ever form dense inputs since we only encode
   * entries that do have any observations.
   * <P/>
   *
   * @param drmC confidence/preference encoded input C* as explained above
   * @param c0 baseline confidence value.
   * @param k factorization rank (~50...100 is probably enough)
   * @param lambda regularization for this iteration
   * @param maxIterations maximum iterations to run
   * @param convergenceTreshold reserved, not used at this point
   */
  def alsImplicit(
      drmC: DrmLike[Int],
      c0: Double = 1.0,
      k: Int = 50,
      lambda: Double = 0.01,
      maxIterations: Int = 10,
      convergenceTreshold: Double = 0.10
      ): ALS.Result[Int] = {
    val drmA = drmC
    val drmAt = drmC.t

    // cbind(U,A):
    var drmUA = drmA.mapBlock(ncol = k + drmA.ncol) {
      case (keys, block) =>
        val uaBlock = block.like(block.nrow, block.ncol + k)
        uaBlock(::, 0 until k) :=
            Matrices.symmetricUniformView(uaBlock.nrow, k, RandomUtils.getRandom().nextInt()) * 0.01
        uaBlock(::, k until uaBlock.ncol) := block
        keys -> uaBlock
    }

    // cbind(V,A'):
    var drmVAt = drmAt.mapBlock(ncol = k + drmAt.ncol) {
      case (keys, block) =>
        val vatBlock = block.like(block.nrow, block.ncol + k)
        vatBlock(::, k until vatBlock.ncol) := block
        keys -> vatBlock
    }
        .checkpoint()

    var i = 0
    var stop = false

    var drmVAtOld: DrmLike[Int] = null
    var drmUAOld: DrmLike[Int] = null

    while (i < maxIterations && !stop) {

      // Update U-A, relinquish stuff explicitly from block manager to alleviate GC concerns and swaps
      if (drmUAOld != null) drmUAOld.uncache()
      drmUAOld = drmUA
      drmUA = updateUA(drmUA, drmVAt, k, c0)

      // Update V-A'
      if (drmVAtOld != null) drmVAtOld.uncache()
      drmVAtOld = drmVAt
      drmVAt = updateUA(drmVAt, drmUA, k, c0)

      i += 1
    }

    new ALS.Result[Int](drmU = drmUA(::, 0 until k), drmV = drmVAt(::, 0 until k), iterationsRMSE = Iterable())
  }

  private def updateUA(drmUA: DrmLike[Int], drmVAt: DrmLike[Int], k: Int, c0: Double): DrmLike[Int] = {

    implicit val ctx = drmUA.context

    val n = drmUA.ncol
    val drmV = drmVAt(::, 0 until k)

    val vtvBcast = drmBroadcast(drmV.t %*% drmV)

    val uaRdd = drmUA.rdd.cogroup(other = generateMessages(drmVAt, k)).map {
      case (uKey, (uavecs, msgs)) =>

        val uavec = uavecs.head
        val urow = uavec(0 until k)
        val arow = uavec(k until n)

        val vsum = new DenseVector(k)
        val m: Matrix = vtvBcast

        msgs.foreach {
          case (vKey, vrow) =>
            val c_u = arow(vKey)

            // (1) if arow[vKey] > 0 means p == 1, in which case we update accumulator vsum.
            if (c_u > 0) vsum += vrow * (c0 + c_u)

            // (2) Update m
            vrow *= abs(c_u)
            m += vrow cross vrow
        }

        // Update u-vec
        urow := solve(a = m, b = vsum)
        uKey -> uavec
    }

    drmWrap(uaRdd, ncol = n)
  }

  // This generates RDD of messages in a form RDD[destVIndex ->(srcUIindex,u-row-vec)]
  private def generateMessages(drmUA: DrmLike[Int], k: Int) = {

    val n = drmUA.ncol
    // Now we delve into Spark-specific processing.
    drmUA.rdd.flatMap {
      case (rowKey, row) =>
        val uvec = new DenseVector(k) := row(0 until k)
        val payload = rowKey -> uvec
        val cvec = row(k until n)
        cvec.nonZeroes().map(_.index() -> payload)
    }
  }

}
