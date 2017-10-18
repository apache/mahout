package scala.org.rawkintrevo.cylon.common.mahout

import org.apache.mahout.math._

import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.MahoutCollections._

import org.apache.mahout.math.algorithms.common.distance._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings.RLikeVectorOps
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.{CardinalityException, Vector}


object DistanceMetricSelector extends Serializable {

  val namedMetricLookup = Map('Chebyshev -> 1.0, 'Cosine -> 2.0, 'Euclidean -> 3.0)

  def select(dm: Double): DistanceMetric = {
    dm match {
      case 1.0 => Chebyshev
      case 2.0 => Cosine
      case 3.0 => Euclidean
    }
  }
}

object Euclidean extends DistanceMetric {
  def distance(v1: Vector, v2: Vector): Double = {
    Math.sqrt(v2.getDistanceSquared(v1))

  }
}
