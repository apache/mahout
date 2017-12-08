package org.apache.mahout.cylon-example.flinkengine.windowfns

import org.apache.flink.api.java.tuple.Tuple
import org.apache.flink.streaming.api.scala.function.WindowFunction
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector

class FilterOutlierClusters(minOccurances: Int = 2) extends WindowFunction[
  (String, DecomposedFace),
  (String, DecomposedFace),
  Tuple,
  TimeWindow] {

  def apply(key: org.apache.flink.api.java.tuple.Tuple,
            window: TimeWindow,
            input: Iterable[(String, DecomposedFace)],
            out: Collector[(String, DecomposedFace)]): Unit = {

    val countsMap = input.toArray.map(t => t._2.cluster).groupBy(identity).mapValues(_.length)
    val dummy =
      input.toArray.filter(t => countsMap(t._2.cluster) >= minOccurances)
        .foreach(out.collect(_))
  }
}
