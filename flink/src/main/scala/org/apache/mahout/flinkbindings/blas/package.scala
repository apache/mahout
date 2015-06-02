package org.apache.mahout.flinkbindings

import org.apache.flink.api.java.functions.KeySelector
import org.apache.mahout.math.Vector
import scala.reflect.ClassTag


package object blas {

  // TODO: remove it once figure out how to make Flink accept interfaces (Vector here)
  def tuple_1[K: ClassTag] = new KeySelector[(Int, K), Integer] {
    def getKey(tuple: Tuple2[Int, K]): Integer = tuple._1
  }

}