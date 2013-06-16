package org.apache.mahout.clustering.streaming.mapreduce;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;

public class StreamingKMeansThread implements Callable<Iterable<Centroid>> {
  private static final int NUM_ESTIMATE_POINTS = 1000;

  private Configuration conf;
  private Iterable<Centroid> datapoints;

  public StreamingKMeansThread(Path input, Configuration conf) {
    this.datapoints = StreamingKMeansUtilsMR.getCentroidsFromVectorWritable(
        new SequenceFileValueIterable<VectorWritable>(input, false, conf));
    this.conf = conf;
  }

  @Override
  public Iterable<Centroid> call() {
    UpdatableSearcher searcher = StreamingKMeansUtilsMR.searcherFromConfiguration(conf);
    int numClusters = conf.getInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS, 1);
    double estimateDistanceCutoff = conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF,
        StreamingKMeansDriver.INVALID_DISTANCE_CUTOFF);

    Iterator<Centroid> datapointsIterator = datapoints.iterator();
    if (estimateDistanceCutoff == StreamingKMeansDriver.INVALID_DISTANCE_CUTOFF) {
      List<Centroid> estimatePoints = Lists.newArrayListWithExpectedSize(NUM_ESTIMATE_POINTS);
      while (datapointsIterator.hasNext() && estimatePoints.size() < NUM_ESTIMATE_POINTS) {
        estimatePoints.add(datapointsIterator.next());
      }
      estimateDistanceCutoff = ClusteringUtils.estimateDistanceCutoff(estimatePoints, searcher.getDistanceMeasure());
    }

    StreamingKMeans clusterer = new StreamingKMeans(searcher, numClusters, estimateDistanceCutoff);
    while (datapointsIterator.hasNext()) {
      clusterer.cluster(datapointsIterator.next());
    }
    clusterer.reindexCentroids();

    return clusterer;
  }

}
