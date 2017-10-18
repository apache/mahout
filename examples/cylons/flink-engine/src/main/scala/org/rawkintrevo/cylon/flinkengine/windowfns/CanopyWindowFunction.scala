package org.rawkintrevo.cylon.flinkengine.windowfns

import org.apache.flink.api.java.tuple.Tuple
import org.apache.flink.streaming.api.functions.co.CoProcessFunction
import org.apache.flink.streaming.api.scala.function.WindowFunction
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.apache.mahout.math.Vector
import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.MahoutCollections._

import scala.collection.JavaConversions._
import org.apache.mahout.math.algorithms.clustering.CanopyFn
import scala.org.rawkintrevo.cylon.common.mahout.DistanceMetricSelector
import org.slf4j.{Logger, LoggerFactory}



class CanopyWindowFunction extends WindowFunction[(String, DecomposedFace),
  (String, Matrix),
  Tuple,
  TimeWindow] {

  val logger: Logger = LoggerFactory.getLogger(classOf[CanopyWindowFunction])

  val dm = DistanceMetricSelector.select(DistanceMetricSelector.namedMetricLookup('Euclidean))
  def apply(key: org.apache.flink.api.java.tuple.Tuple,
            window: TimeWindow,
            input: Iterable[(String, DecomposedFace)],
            out: Collector[(String, Matrix)]) {

    val t1 = input.toArray.map(t => Math.min(t._2.w, t._2.h)).min
    val t2 = input.toArray.map(t => Math.max(t._2.w, t._2.h)).max
    val incoreMat = dense(input.toArray.map(t => t._2.metaVec))
    val centers: Matrix = CanopyFn.findCenters(incoreMat, dm, t1, t2)

    out.collect((key.getField(0), centers))
  }
}





class CanopyAssignmentCoProcessFunction extends CoProcessFunction[
  DecomposedFace,
  (String, Matrix),
  (String, DecomposedFace)] {

  var workingCanopyMatrix: Option[Matrix] = None
  var canopyMatricesRecieved = 0
  val dm = DistanceMetricSelector.select(DistanceMetricSelector.namedMetricLookup('Euclidean))
  def processElement1(in1: DecomposedFace,
    context: CoProcessFunction[DecomposedFace, (String, Matrix), (String, DecomposedFace)]#Context,
    collector: Collector[(String, DecomposedFace)]): Unit = {

    val newCluster = workingCanopyMatrix match {
      case Some(m) => {
        val cluster: Int = (0 until m.nrow).foldLeft(-1, 9999999999999999.9)((l, r) => {
          val dist = dm.distance(m(r, ::), in1.metaVec)
          if ((dist) < l._2) {
            (r, dist)
          }
          else {
            l
          }
        })._1
        cluster
      }
      case None => -1  // If the workingCanopyMatrix isn't warmed up yet- then assign everything to -1 cluster.
    }

    collector.collect((in1.key, in1.copy(cluster = newCluster)))
  }

  def processElement2(in2: (String, Matrix),
     context: CoProcessFunction[DecomposedFace, (String, Matrix), (String, DecomposedFace)]#Context,
     collector: Collector[(String, DecomposedFace)]): Unit = {
    workingCanopyMatrix = Some(in2._2)
    canopyMatricesRecieved += 1
    //collector.collect("foo", DecomposedFace("foo", 0,0,0,0,0,dvec(0)), -1)
  }

}