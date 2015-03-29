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

package org.apache.mahout.clustering.topdown.postprocessor;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;

public final class ClusterCountReaderTest extends MahoutTestCase {
  
  public static final double[][] REFERENCE = { {1, 1}, {2, 1}, {1, 2}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};
  
  private FileSystem fs;
  private Path outputPathForCanopy;
  private Path outputPathForKMeans;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = getConfiguration();
    fs = FileSystem.get(conf);
  }
  
  public static List<VectorWritable> getPointsWritable(double[][] raw) {
    List<VectorWritable> points = Lists.newArrayList();
    for (double[] fr : raw) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }
  
  /**
   * Story: User wants to use cluster post processor after canopy clustering and then run clustering on the
   * output clusters
   */
  @Test
  public void testGetNumberOfClusters() throws Exception {
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    
    Path pointsPath = getTestTempDirPath("points");
    Configuration conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file2"), fs, conf);
    
    outputPathForCanopy = getTestTempDirPath("canopy");
    outputPathForKMeans = getTestTempDirPath("kmeans");
    
    topLevelClustering(pointsPath, conf);
    
    int numberOfClusters = ClusterCountReader.getNumberOfClusters(outputPathForKMeans, conf);
    Assert.assertEquals(2, numberOfClusters);
    verifyThatNumberOfClustersIsCorrect(conf, new Path(outputPathForKMeans, new Path("clusteredPoints")));
    
  }
  
  private void topLevelClustering(Path pointsPath, Configuration conf) throws IOException,
                                                                      InterruptedException,
                                                                      ClassNotFoundException {
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    CanopyDriver.run(conf, pointsPath, outputPathForCanopy, measure, 4.0, 3.0, true, 0.0, true);
    Path clustersIn = new Path(outputPathForCanopy, new Path(Cluster.CLUSTERS_DIR + '0'
                                                                   + Cluster.FINAL_ITERATION_SUFFIX));
    KMeansDriver.run(conf, pointsPath, clustersIn, outputPathForKMeans, 1, 1, true, 0.0, true);
  }
  
  private static void verifyThatNumberOfClustersIsCorrect(Configuration conf, Path clusteredPointsPath) {
    DummyOutputCollector<IntWritable,WeightedVectorWritable> collector =
        new DummyOutputCollector<IntWritable,WeightedVectorWritable>();
    
    // The key is the clusterId, the value is the weighted vector
    for (Pair<IntWritable,WeightedVectorWritable> record :
         new SequenceFileIterable<IntWritable,WeightedVectorWritable>(new Path(clusteredPointsPath, "part-m-0"),
                                                                      conf)) {
      collector.collect(record.getFirst(), record.getSecond());
    }
    int clusterSize = collector.getKeys().size();
    assertEquals(2, clusterSize);
  }
  
}
