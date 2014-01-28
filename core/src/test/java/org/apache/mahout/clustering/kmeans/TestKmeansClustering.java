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

package org.apache.mahout.clustering.kmeans;

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

public final class TestKmeansClustering extends MahoutTestCase {
  
  public static final double[][] REFERENCE = { {1, 1}, {2, 1}, {1, 2}, {2, 2}, {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};
  
  private static final int[][] EXPECTED_NUM_POINTS = { {9}, {4, 5}, {4, 4, 1}, {1, 2, 1, 5}, {1, 1, 1, 2, 4},
      {1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 2, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
  
  private FileSystem fs;
  
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
  
  public static List<VectorWritable> getPointsWritableDenseVector(double[][] raw) {
    List<VectorWritable> points = Lists.newArrayList();
    for (double[] fr : raw) {
      Vector vec = new DenseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }
  
  public static List<Vector> getPoints(double[][] raw) {
    List<Vector> points = Lists.newArrayList();
    for (double[] fr : raw) {
      Vector vec = new SequentialAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }
  
  /**
   * Tests
   * {@link KMeansClusterer#runKMeansIteration(Iterable, Iterable, DistanceMeasure, double)}
   * ) single run convergence with a given distance threshold.
   */
  /*@Test
  public void testRunKMeansIterationConvergesInOneRunWithGivenDistanceThreshold() {
    double[][] rawPoints = { {0, 0}, {0, 0.25}, {0, 0.75}, {0, 1}};
    List<Vector> points = getPoints(rawPoints);

    ManhattanDistanceMeasure distanceMeasure = new ManhattanDistanceMeasure();
    List<Kluster> clusters = Arrays.asList(new Kluster(points.get(0), 0, distanceMeasure), new Kluster(points.get(3),
        3, distanceMeasure));

    // To converge in a single run, the given distance threshold should be
    // greater than or equal to 0.125,
    // since 0.125 will be the distance between center and centroid for the
    // initial two clusters after one run.
    double distanceThreshold = 0.25;

    boolean converged = KMeansClusterer.runKMeansIteration(points, clusters, distanceMeasure, distanceThreshold);

    Vector cluster1Center = clusters.get(0).getCenter();
    assertEquals(0, cluster1Center.get(0), EPSILON);
    assertEquals(0.125, cluster1Center.get(1), EPSILON);

    Vector cluster2Center = clusters.get(1).getCenter();
    assertEquals(0, cluster2Center.get(0), EPSILON);
    assertEquals(0.875, cluster2Center.get(1), EPSILON);

    assertTrue("KMeans iteration should be converged after a single run", converged);
  }*/

  /** Story: User wishes to run kmeans job on reference data */
  @Test
  public void testKMeansSeqJob() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    
    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file2"), fs, conf);
    for (int k = 1; k < points.size(); k++) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Path path = new Path(clustersPath, "part-00000");
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Kluster.class);
      try {
        for (int i = 0; i < k + 1; i++) {
          Vector vec = points.get(i).get();
          
          Kluster cluster = new Kluster(vec, i, measure);
          // add the center so the centroid will be correct upon output
          cluster.observe(cluster.getCenter(), 1);
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.close(writer, false);
      }
      // now run the Job
      Path outputPath = getTestTempDirPath("output");
      String[] args = {optKey(DefaultOptionCreator.INPUT_OPTION), pointsPath.toString(),
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION), clustersPath.toString(),
          optKey(DefaultOptionCreator.OUTPUT_OPTION), outputPath.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.001",
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "2", optKey(DefaultOptionCreator.CLUSTERING_OPTION),
          optKey(DefaultOptionCreator.OVERWRITE_OPTION), optKey(DefaultOptionCreator.METHOD_OPTION),
          DefaultOptionCreator.SEQUENTIAL_METHOD};
      ToolRunner.run(conf, new KMeansDriver(), args);
      
      // now compare the expected clusters with actual
      Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
      int[] expect = EXPECTED_NUM_POINTS[k];
      DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable> collector = new DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable>();
      // The key is the clusterId, the value is the weighted vector
      for (Pair<IntWritable,WeightedPropertyVectorWritable> record : new SequenceFileIterable<IntWritable,WeightedPropertyVectorWritable>(
          new Path(clusteredPointsPath, "part-m-0"), conf)) {
        collector.collect(record.getFirst(), record.getSecond());
      }
      assertEquals("clusters[" + k + ']', expect.length, collector.getKeys().size());
    }
  }
  
  /** Story: User wishes to run kmeans job on reference data (DenseVector test) */
  @Test
  public void testKMeansSeqJobDenseVector() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    List<VectorWritable> points = getPointsWritableDenseVector(REFERENCE);
    
    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file2"), fs, conf);
    for (int k = 1; k < points.size(); k++) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Path path = new Path(clustersPath, "part-00000");
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Kluster.class);
      try {
        for (int i = 0; i < k + 1; i++) {
          Vector vec = points.get(i).get();
          
          Kluster cluster = new Kluster(vec, i, measure);
          // add the center so the centroid will be correct upon output
          cluster.observe(cluster.getCenter(), 1);
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.close(writer, false);
      }
      // now run the Job
      Path outputPath = getTestTempDirPath("output");
      String[] args = {optKey(DefaultOptionCreator.INPUT_OPTION), pointsPath.toString(),
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION), clustersPath.toString(),
          optKey(DefaultOptionCreator.OUTPUT_OPTION), outputPath.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.001",
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "2", optKey(DefaultOptionCreator.CLUSTERING_OPTION),
          optKey(DefaultOptionCreator.OVERWRITE_OPTION), optKey(DefaultOptionCreator.METHOD_OPTION),
          DefaultOptionCreator.SEQUENTIAL_METHOD};
      ToolRunner.run(conf, new KMeansDriver(), args);
      
      // now compare the expected clusters with actual
      Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
      int[] expect = EXPECTED_NUM_POINTS[k];
      DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable> collector = new DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable>();
      // The key is the clusterId, the value is the weighted vector
      for (Pair<IntWritable,WeightedPropertyVectorWritable> record : new SequenceFileIterable<IntWritable,WeightedPropertyVectorWritable>(
          new Path(clusteredPointsPath, "part-m-0"), conf)) {
        collector.collect(record.getFirst(), record.getSecond());
      }
      assertEquals("clusters[" + k + ']', expect.length, collector.getKeys().size());
    }
  }
  
  /** Story: User wishes to run kmeans job on reference data */
  @Test
  public void testKMeansMRJob() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    
    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file2"), fs, conf);
    for (int k = 1; k < points.size(); k += 3) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Path path = new Path(clustersPath, "part-00000");
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Kluster.class);
      
      try {
        for (int i = 0; i < k + 1; i++) {
          Vector vec = points.get(i).get();
          
          Kluster cluster = new Kluster(vec, i, measure);
          // add the center so the centroid will be correct upon output
          cluster.observe(cluster.getCenter(), 1);
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.close(writer, false);
      }
      // now run the Job
      Path outputPath = getTestTempDirPath("output");
      String[] args = {optKey(DefaultOptionCreator.INPUT_OPTION), pointsPath.toString(),
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION), clustersPath.toString(),
          optKey(DefaultOptionCreator.OUTPUT_OPTION), outputPath.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.001",
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "2", optKey(DefaultOptionCreator.CLUSTERING_OPTION),
          optKey(DefaultOptionCreator.OVERWRITE_OPTION)};
      ToolRunner.run(getConfiguration(), new KMeansDriver(), args);
      
      // now compare the expected clusters with actual
      Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
      // assertEquals("output dir files?", 4, outFiles.length);
      int[] expect = EXPECTED_NUM_POINTS[k];
      DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable> collector = new DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable>();
      // The key is the clusterId, the value is the weighted vector
      for (Pair<IntWritable,WeightedPropertyVectorWritable> record : new SequenceFileIterable<IntWritable,WeightedPropertyVectorWritable>(
          new Path(clusteredPointsPath, "part-m-00000"), conf)) {
        collector.collect(record.getFirst(), record.getSecond());
      }
      assertEquals("clusters[" + k + ']', expect.length, collector.getKeys().size());
    }
  }
  
  /**
   * Story: User wants to use canopy clustering to input the initial clusters
   * for kmeans job.
   */
  @Test
  public void testKMeansWithCanopyClusterInput() throws Exception {
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    
    Path pointsPath = getTestTempDirPath("points");
    Configuration conf = getConfiguration();
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, true, new Path(pointsPath, "file2"), fs, conf);
    
    Path outputPath = getTestTempDirPath("output");
    // now run the Canopy job
    CanopyDriver.run(conf, pointsPath, outputPath, new ManhattanDistanceMeasure(), 3.1, 2.1, false, 0.0, false);
    
    DummyOutputCollector<Text, ClusterWritable> collector1 =
        new DummyOutputCollector<Text, ClusterWritable>();

    FileStatus[] outParts = FileSystem.get(conf).globStatus(
                    new Path(outputPath, "clusters-0-final/*-0*"));
    for (FileStatus outPartStat : outParts) {
      for (Pair<Text,ClusterWritable> record :
               new SequenceFileIterable<Text,ClusterWritable>(
                 outPartStat.getPath(), conf)) {
          collector1.collect(record.getFirst(), record.getSecond());
      }
    }

    boolean got15 = false;
    boolean got43 = false;
    int count = 0;
    for (Text k : collector1.getKeys()) {
      count++;
      List<ClusterWritable> vl = collector1.getValue(k);
      assertEquals("non-singleton centroid!", 1, vl.size());
      ClusterWritable clusterWritable = vl.get(0);
      Vector v = clusterWritable.getValue().getCenter();
      assertEquals("cetriod vector is wrong length", 2, v.size());
      if ( (Math.abs(v.get(0) - 1.5) < EPSILON) 
                  && (Math.abs(v.get(1) - 1.5) < EPSILON)
                  && !got15) {
        got15 = true;
      } else if ( (Math.abs(v.get(0) - 4.333333333333334) < EPSILON) 
                  && (Math.abs(v.get(1) - 4.333333333333334) < EPSILON)
                  && !got43) {
        got43 = true;
      } else {
        fail("got unexpected center: " + v + " [" + v.getClass().toString() + ']');
      }
    }
    assertEquals("got unexpected number of centers", 2, count);

    // now run the KMeans job
    Path kmeansOutput = new Path(outputPath, "kmeans");
	  KMeansDriver.run(getConfiguration(), pointsPath, new Path(outputPath, "clusters-0-final"), kmeansOutput,
      0.001, 10, true, 0.0, false);
    
    // now compare the expected clusters with actual
    Path clusteredPointsPath = new Path(kmeansOutput, "clusteredPoints");
    DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable> collector = new DummyOutputCollector<IntWritable,WeightedPropertyVectorWritable>();
    
    // The key is the clusterId, the value is the weighted vector
    for (Pair<IntWritable,WeightedPropertyVectorWritable> record : new SequenceFileIterable<IntWritable,WeightedPropertyVectorWritable>(
        new Path(clusteredPointsPath, "part-m-00000"), conf)) {
      collector.collect(record.getFirst(), record.getSecond());
    }
    
    for (IntWritable k : collector.getKeys()) {
      List<WeightedPropertyVectorWritable> wpvList = collector.getValue(k);
      assertTrue("empty cluster!", !wpvList.isEmpty());
      if (wpvList.get(0).getVector().get(0) <= 2.0) {
        for (WeightedPropertyVectorWritable wv : wpvList) {
          Vector v = wv.getVector();
          int idx = v.maxValueIndex();
          assertTrue("bad cluster!", v.get(idx) <= 2.0);
        }
        assertEquals("Wrong size cluster", 4, wpvList.size());
      } else {
        for (WeightedPropertyVectorWritable wv : wpvList) {
          Vector v = wv.getVector();
          int idx = v.minValueIndex();
          assertTrue("bad cluster!", v.get(idx) > 2.0);
        }
        assertEquals("Wrong size cluster", 5, wpvList.size());
      }
    }
  }
}
