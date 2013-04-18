package org.apache.mahout.benchmark;

import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

public class ClosestCentroidBenchmark {
  public static final String SERIALIZE = "Serialize";
  public static final String DESERIALIZE = "Deserialize";
  private final VectorBenchmarks mark;

  public ClosestCentroidBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark(DistanceMeasure measure) throws IOException {
    SparseMatrix clusterDistances = new SparseMatrix(mark.numClusters, mark.numClusters);
    for (int i = 0; i < mark.numClusters; i++) {
      for (int j = 0; j < mark.numClusters; j++) {
        double distance = Double.POSITIVE_INFINITY;
        if (i != j) {
          distance = measure.distance(mark.clusters[i], mark.clusters[j]);
        }
        clusterDistances.setQuick(i, j, distance);
      }
    }

    long distanceCalculations = 0;
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < mark.loop; l++) {
      TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
      for (int i = 0; i < mark.numVectors; i++) {
        Vector vector = mark.vectors[1][mark.vIndex(i)];
        double minDistance = Double.MAX_VALUE;
        for (int k = 0; k < mark.numClusters; k++) {
          double distance = measure.distance(vector, mark.clusters[k]);
          distanceCalculations++;
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
      }
      if (call.end(mark.maxTimeUsec)) {
        break;
      }
    }
    mark.printStats(stats, measure.getClass().getName(), "Closest C w/o Elkan's trick", "distanceCalculations = "
        + distanceCalculations);

    distanceCalculations = 0;
    stats = new TimingStatistics();
    Random rand = RandomUtils.getRandom();
    for (int l = 0; l < mark.loop; l++) {
      TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
      for (int i = 0; i < mark.numVectors; i++) {
        Vector vector = mark.vectors[1][mark.vIndex(i)];
        int closestCentroid = rand.nextInt(mark.numClusters);
        double dist = measure.distance(vector, mark.clusters[closestCentroid]);
        distanceCalculations++;
        for (int k = 0; k < mark.numClusters; k++) {
          if (closestCentroid != k) {
            double centroidDist = clusterDistances.getQuick(k, closestCentroid);
            if (centroidDist < 2 * dist) {
              dist = measure.distance(vector, mark.clusters[k]);
              closestCentroid = k;
              distanceCalculations++;
            }
          }
        }
      }
      if (call.end(mark.maxTimeUsec)) {
        break;
      }
    }
    mark.printStats(stats, measure.getClass().getName(), "Closest C w/ Elkan's trick", "distanceCalculations = "
        + distanceCalculations);
  }
}
