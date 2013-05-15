package org.apache.mahout.clustering.streaming.mapreduce;

import java.util.concurrent.Callable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;

public class StreamingKMeansThread implements Callable<Iterable<Centroid>> {
  private Configuration conf;
  private Iterable<Centroid> datapoints;

  public StreamingKMeansThread(Path input, Configuration conf) {
    this.datapoints = StreamingKMeansUtilsMR.getCentroidsFromVectorWritable(new SequenceFileValueIterable<VectorWritable>(input, false, conf));
    this.conf = conf;
  }

  public StreamingKMeansThread(Iterable<Centroid> datapoints, Configuration conf) {
    this.datapoints = datapoints;
    this.conf = conf;
  }

  @Override
  public Iterable<Centroid> call() throws Exception {
    UpdatableSearcher searcher = StreamingKMeansUtilsMR.searcherFromConfiguration(conf);
    int numClusters = conf.getInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS, 1);

    double estimateDistanceCutoff = conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF,
        (float) ClusteringUtils.estimateDistanceCutoff(datapoints, searcher.getDistanceMeasure(), 100));

    StreamingKMeans clusterer = new StreamingKMeans(searcher, numClusters, estimateDistanceCutoff);
    clusterer.cluster(datapoints);
    clusterer.reindexCentroids();

    return clusterer;
  }

}
