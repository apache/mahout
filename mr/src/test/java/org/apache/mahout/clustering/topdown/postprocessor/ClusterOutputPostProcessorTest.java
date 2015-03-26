/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.clustering.topdown.postprocessor;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.topdown.PathDirectory;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;

public final class ClusterOutputPostProcessorTest extends MahoutTestCase {

  private static final double[][] REFERENCE = { {1, 1}, {2, 1}, {1, 2}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};

  private FileSystem fs;

  private Path outputPath;

  private Configuration conf;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = getConfiguration();
    fs = FileSystem.get(conf);
  }

  private static List<VectorWritable> getPointsWritable(double[][] raw) {
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
  public void testTopDownClustering() throws Exception {
    List<VectorWritable> points = getPointsWritable(REFERENCE);

    Path pointsPath = getTestTempDirPath("points");
    conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file2"), fs, conf);

    outputPath = getTestTempDirPath("output");

    topLevelClustering(pointsPath, conf);

    Map<String,Path> postProcessedClusterDirectories = ouputPostProcessing(conf);

    assertPostProcessedOutput(postProcessedClusterDirectories);

    bottomLevelClustering(postProcessedClusterDirectories);
  }

  private void assertTopLevelCluster(Entry<String,Path> cluster) {
    String clusterId = cluster.getKey();
    Path clusterPath = cluster.getValue();

    try {
      if ("0".equals(clusterId)) {
        assertPointsInFirstTopLevelCluster(clusterPath);
      } else if ("1".equals(clusterId)) {
        assertPointsInSecondTopLevelCluster(clusterPath);
      }
    } catch (IOException e) {
      Assert.fail("Exception occurred while asserting top level cluster.");
    }

  }

  private void assertPointsInFirstTopLevelCluster(Path clusterPath) throws IOException {
    List<Vector> vectorsInCluster = getVectorsInCluster(clusterPath);
    for (Vector vector : vectorsInCluster) {
      Assert.assertTrue(ArrayUtils.contains(new String[] {"{0:1.0,1:1.0}", "{0:2.0,1:1.0}", "{0:1.0,1:2.0}"},
        vector.asFormatString()));
    }
  }

  private void assertPointsInSecondTopLevelCluster(Path clusterPath) throws IOException {
    List<Vector> vectorsInCluster = getVectorsInCluster(clusterPath);
    for (Vector vector : vectorsInCluster) {
      Assert.assertTrue(ArrayUtils.contains(new String[] {"{0:4.0,1:4.0}", "{0:5.0,1:4.0}", "{0:4.0,1:5.0}",
                                                          "{0:5.0,1:5.0}"}, vector.asFormatString()));
    }
  }

  private List<Vector> getVectorsInCluster(Path clusterPath) throws IOException {
    Path[] partFilePaths = FileUtil.stat2Paths(fs.globStatus(clusterPath));
    FileStatus[] listStatus = fs.listStatus(partFilePaths);
    List<Vector> vectors = Lists.newArrayList();
    for (FileStatus partFile : listStatus) {
      SequenceFile.Reader topLevelClusterReader = new SequenceFile.Reader(fs, partFile.getPath(), conf);
      Writable clusterIdAsKey = new LongWritable();
      VectorWritable point = new VectorWritable();
      while (topLevelClusterReader.next(clusterIdAsKey, point)) {
        vectors.add(point.get());
      }
    }
    return vectors;
  }

  private void bottomLevelClustering(Map<String,Path> postProcessedClusterDirectories) throws IOException,
                                                                                      InterruptedException,
                                                                                      ClassNotFoundException {
    for (Entry<String,Path> topLevelCluster : postProcessedClusterDirectories.entrySet()) {
      String clusterId = topLevelCluster.getKey();
      Path topLevelclusterPath = topLevelCluster.getValue();

      Path bottomLevelCluster = PathDirectory.getBottomLevelClusterPath(outputPath, clusterId);
      CanopyDriver.run(conf, topLevelclusterPath, bottomLevelCluster, new ManhattanDistanceMeasure(), 2.1,
        2.0, true, 0.0, true);
      assertBottomLevelCluster(bottomLevelCluster);
    }
  }

  private void assertBottomLevelCluster(Path bottomLevelCluster) {
    Path clusteredPointsPath = new Path(bottomLevelCluster, "clusteredPoints");

    DummyOutputCollector<IntWritable,WeightedVectorWritable> collector =
        new DummyOutputCollector<IntWritable,WeightedVectorWritable>();

    // The key is the clusterId, the value is the weighted vector
    for (Pair<IntWritable,WeightedVectorWritable> record :
         new SequenceFileIterable<IntWritable,WeightedVectorWritable>(new Path(clusteredPointsPath, "part-m-0"),
                                                                      conf)) {
      collector.collect(record.getFirst(), record.getSecond());
    }
    int clusterSize = collector.getKeys().size();
    // First top level cluster produces two more clusters, second top level cluster is not broken again
    assertTrue(clusterSize == 1 || clusterSize == 2);

  }

  private void assertPostProcessedOutput(Map<String,Path> postProcessedClusterDirectories) {
    for (Entry<String,Path> cluster : postProcessedClusterDirectories.entrySet()) {
      assertTopLevelCluster(cluster);
    }
  }

  private Map<String,Path> ouputPostProcessing(Configuration conf) throws IOException {
    ClusterOutputPostProcessor clusterOutputPostProcessor = new ClusterOutputPostProcessor(outputPath,
        outputPath, conf);
    clusterOutputPostProcessor.process();
    return clusterOutputPostProcessor.getPostProcessedClusterDirectories();
  }

  private void topLevelClustering(Path pointsPath, Configuration conf) throws IOException,
                                                                      InterruptedException,
                                                                      ClassNotFoundException {
    CanopyDriver.run(conf, pointsPath, outputPath, new ManhattanDistanceMeasure(), 3.1, 2.1, true, 0.0, true);
  }

}
