package org.rawkintrevo.cylon.flinkengine.windowfns

import org.apache.flink.api.java.tuple.Tuple
import org.apache.flink.streaming.api.scala.function.WindowFunction
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.apache.mahout.math.algorithms.clustering.CanopyFn
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings.{::, dense}
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.solr.client.solrj.response.QueryResponse
import org.apache.solr.common.SolrDocument
import org.rawkintrevo.cylon.common.solr.CylonSolrClient

import scala.org.rawkintrevo.cylon.common.mahout.DistanceMetricSelector

class SolrLookupWindowFunction(solrURL :String,
                               minOccurances: Int = 2,
                               newFaceDistanceThreshold: Double = 2000.0
                                     ) extends WindowFunction[(String, DecomposedFace),
  (String, DecomposedFace),
  Tuple,
  TimeWindow] {


    val dm = DistanceMetricSelector.select(DistanceMetricSelector.namedMetricLookup('Euclidean))

    var seenFacesMap: scala.collection.mutable.Map[Int, DecomposedFace] = scala.collection.mutable.Map()

    var frames: Array[Int] = Array(0)


    def apply(key: org.apache.flink.api.java.tuple.Tuple,
              window: TimeWindow,
              input: Iterable[(String, DecomposedFace)],
              out: Collector[(String, DecomposedFace)]) {

      val inputArray = input.toArray

//      // Step 2: Filter Ghosts
      val countsMap = inputArray.map(t => t._2.cluster).groupBy(identity).mapValues(_.length)
      val filteredInputArray: Array[(String,
                                    DecomposedFace)] = inputArray.filter(t => countsMap(t._2.cluster) >= minOccurances)

      //** Better strategy right here **.

      // Step 3a: Get "Average" of Rects (or mean smooth them some other way)
      // (Or maybe search Solr w/ea image- get average of returned ranks
      val solrClient = new CylonSolrClient
      solrClient.connect(solrURL)

       val queryArray: Array[(Int, Array[SolrDocument])] = filteredInputArray
            .groupBy(_._2.cluster)
            .toArray
            .map(t => (t._1, dense(t._2.map(_._2.v)).colMeans))
            .map(t => (t._1, solrClient.getDocsArray(solrClient.eigenFaceQuery(t._2))))

      val clusterNames: Array[(Int, Array[(String, Double)])] =   queryArray
        .map(t => (t._1, t._2.map(d => (d.get("name_s").asInstanceOf[String], d.get("calc_dist").asInstanceOf[Double]))
                              .sortBy(_._2))) // Now we have a list of humans sorted in ascending order based on distance.
      // (cluster, Array[(Name, Distance)])


      val nameClusterMap = clusterNames.map(t => {
        var bestMatch = ("unidentified", 0.0)
        if (t._2.nonEmpty) {
          if (t._2(0)._2 < newFaceDistanceThreshold){
            bestMatch = t._2(0)
          } else {
            bestMatch = ("unidentified", t._2(0)._2)
          }
        }
        (t._1, bestMatch)
      }).toMap



      // 4) Make decision to add face or recognize person from results
      // 5) Emit Records
      val new_frames: Array[Int] = input.toArray.map(t => t._2.frame)
      val frames_to_emit = new_frames.diff(frames)
      frames = new_frames


      inputArray.filter(t => frames_to_emit.contains(t._2.frame))
          .map(t => {
            val clusterNum = t._2.cluster
            var newName = t._2.name
            var bestDist = t._2.distanceFromCenter

            if (nameClusterMap.contains(clusterNum)){
              newName = nameClusterMap(clusterNum)._1
              bestDist = nameClusterMap(clusterNum)._2
            }
            if (newName == "unidentified") {
              newName = "human-" + scala.util.Random.alphanumeric.take(5).mkString("").toUpperCase
              inputArray.filter(_._2 == clusterNum).foreach(
                t => solrClient.addFaceToSolr(t._2.v, newName)
              )
              solrClient.commit()
            }
            (t._1, t._2.copy(name= newName, distanceFromCenter = bestDist))
          })
          .map(t => out.collect(t))
      solrClient.solrClient.close()
    }

  def assignVectorToCluster(v: Vector, centers: Matrix): Double = {
    val cluster: Int = (0 until centers.numRows()).foldLeft(-1, 9999999999999999.9)((l, r) => {
      val dist = dm.distance(centers(r, ::), v)
      if ((dist) < l._2) {
        (r, dist)
      }
      else {
        l
      }
    })._1
    cluster
  }
}

