/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.streaming.mapreduce;

import java.io.IOException;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.streaming.cluster.BallKMeans;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StreamingKMeansReducer extends Reducer<IntWritable, CentroidWritable, IntWritable, CentroidWritable> {

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansReducer.class);

  /**
   * Configuration for the MapReduce job.
   */
  private Configuration conf;

  @Override
  public void setup(Context context) {
    // At this point the configuration received from the Driver is assumed to be valid.
    // No other checks are made.
    conf = context.getConfiguration();
  }

  @Override
  public void reduce(IntWritable key, Iterable<CentroidWritable> centroids,
                     Context context) throws IOException, InterruptedException {
    List<Centroid> intermediateCentroids;
    // There might be too many intermediate centroids to fit into memory, in which case, we run another pass
    // of StreamingKMeans to collapse the clusters further.
    if (conf.getBoolean(StreamingKMeansDriver.REDUCE_STREAMING_KMEANS, false)) {
      intermediateCentroids = Lists.newArrayList(
          new StreamingKMeansThread(Iterables.transform(centroids, new Function<CentroidWritable, Centroid>() {
            @Override
            public Centroid apply(CentroidWritable input) {
              Preconditions.checkNotNull(input);
              return input.getCentroid().clone();
            }
          }), conf).call());
    } else {
      intermediateCentroids = centroidWritablesToList(centroids);
    }

    int index = 0;
    for (Vector centroid : getBestCentroids(intermediateCentroids, conf)) {
      context.write(new IntWritable(index), new CentroidWritable((Centroid) centroid));
      ++index;
    }
  }

  public static List<Centroid> centroidWritablesToList(Iterable<CentroidWritable> centroids) {
    // A new list must be created because Hadoop iterators mutate the contents of the Writable in
    // place, without allocating new references when iterating through the centroids Iterable.
    return Lists.newArrayList(Iterables.transform(centroids, new Function<CentroidWritable, Centroid>() {
      @Override
      public Centroid apply(CentroidWritable input) {
        Preconditions.checkNotNull(input);
        return input.getCentroid().clone();
      }
    }));
  }

  public static Iterable<Vector> getBestCentroids(List<Centroid> centroids, Configuration conf) {

    if (log.isInfoEnabled()) {
      log.info("Number of Centroids: {}", centroids.size());
    }

    int numClusters = conf.getInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 1);
    int maxNumIterations = conf.getInt(StreamingKMeansDriver.MAX_NUM_ITERATIONS, 10);
    float trimFraction = conf.getFloat(StreamingKMeansDriver.TRIM_FRACTION, 0.9f);
    boolean kMeansPlusPlusInit = !conf.getBoolean(StreamingKMeansDriver.RANDOM_INIT, false);
    boolean correctWeights = !conf.getBoolean(StreamingKMeansDriver.IGNORE_WEIGHTS, false);
    float testProbability = conf.getFloat(StreamingKMeansDriver.TEST_PROBABILITY, 0.1f);
    int numRuns = conf.getInt(StreamingKMeansDriver.NUM_BALLKMEANS_RUNS, 3);

    BallKMeans ballKMeansCluster = new BallKMeans(StreamingKMeansUtilsMR.searcherFromConfiguration(conf),
        numClusters, maxNumIterations, trimFraction, kMeansPlusPlusInit, correctWeights, testProbability, numRuns);
    return ballKMeansCluster.cluster(centroids);
  }
}
