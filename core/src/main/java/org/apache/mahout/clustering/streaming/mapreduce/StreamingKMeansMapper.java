/*
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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;

public class StreamingKMeansMapper extends Mapper<Writable, VectorWritable, IntWritable, CentroidWritable> {
  /**
   * The clusterer object used to cluster the points received by this mapper online.
   */
  private StreamingKMeans clusterer;

  /**
   * Number of points clustered so far.
   */
  private int numPoints = 0;

  @Override
  public void setup(Context context) {
    // At this point the configuration received from the Driver is assumed to be valid.
    // No other checks are made.
    Configuration conf = context.getConfiguration();
    UpdatableSearcher searcher = StreamingKMeansUtilsMR.searcherFromConfiguration(conf);
    int numClusters = conf.getInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS, 1);
    // There is no way of estimating the distance cutoff unless we have some data.
    clusterer = new StreamingKMeans(searcher, numClusters,
        conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF, 1e-4f));
  }

  @Override
  public void map(Writable key, VectorWritable point, Context context) {
    clusterer.cluster(new Centroid(numPoints++, point.get().clone(), 1));
  }

  @Override
  public void cleanup(Context context) throws IOException, InterruptedException {
    // Reindex the centroids before passing them to the reducer.
    clusterer.reindexCentroids();
    // All outputs have the same key to go to the same final reducer.
    for (Centroid centroid : clusterer) {
      context.write(new IntWritable(0), new CentroidWritable(centroid));
    }
  }
}
