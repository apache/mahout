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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StreamingKMeansThread implements Callable<Iterable<Centroid>> {
  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansThread.class);

  private static final int NUM_ESTIMATE_POINTS = 1000;

  private final Configuration conf;
  private final Iterable<Centroid> dataPoints;

  public StreamingKMeansThread(Path input, Configuration conf) {
    this(StreamingKMeansUtilsMR.getCentroidsFromVectorWritable(
        new SequenceFileValueIterable<VectorWritable>(input, false, conf)), conf);
  }

  public StreamingKMeansThread(Iterable<Centroid> dataPoints, Configuration conf) {
    this.dataPoints = dataPoints;
    this.conf = conf;
  }

  @Override
  public Iterable<Centroid> call() {
    UpdatableSearcher searcher = StreamingKMeansUtilsMR.searcherFromConfiguration(conf);
    int numClusters = conf.getInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS, 1);
    double estimateDistanceCutoff = conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF,
        StreamingKMeansDriver.INVALID_DISTANCE_CUTOFF);

    Iterator<Centroid> dataPointsIterator = dataPoints.iterator();

    if (estimateDistanceCutoff == StreamingKMeansDriver.INVALID_DISTANCE_CUTOFF) {
      List<Centroid> estimatePoints = Lists.newArrayListWithExpectedSize(NUM_ESTIMATE_POINTS);
      while (dataPointsIterator.hasNext() && estimatePoints.size() < NUM_ESTIMATE_POINTS) {
        Centroid centroid = dataPointsIterator.next();
        estimatePoints.add(centroid);
      }

      if (log.isInfoEnabled()) {
        log.info("Estimated Points: {}", estimatePoints.size());
      }
      estimateDistanceCutoff = ClusteringUtils.estimateDistanceCutoff(estimatePoints, searcher.getDistanceMeasure());
    }

    StreamingKMeans streamingKMeans = new StreamingKMeans(searcher, numClusters, estimateDistanceCutoff);

    // datapointsIterator could be empty if no estimate distance was initially provided
    // hence creating the iterator again here for the clustering
    if (!dataPointsIterator.hasNext()) {
      dataPointsIterator = dataPoints.iterator();
    }

    while (dataPointsIterator.hasNext()) {
      streamingKMeans.cluster(dataPointsIterator.next());
    }

    streamingKMeans.reindexCentroids();
    return streamingKMeans;
  }

}
