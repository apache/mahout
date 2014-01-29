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

package org.apache.mahout.clustering;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.evaluation.ClusterEvaluator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public final class TestClusterEvaluator extends MahoutTestCase {
  
  private static final double[][] REFERENCE = { {1, 1}, {2, 1}, {1, 2}, {2, 2}, {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};
  
  private List<VectorWritable> referenceData = Lists.newArrayList();
  
  private final List<VectorWritable> sampleData = Lists.newArrayList();
  
  private Map<Integer,List<VectorWritable>> representativePoints;
  
  private List<Cluster> clusters;
  
  private static final Logger log = LoggerFactory.getLogger(TestClusterEvaluator.class);
  
  private Configuration conf;
  
  private FileSystem fs;
  
  private Path testdata;
  
  private Path output;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = getConfiguration();
    fs = FileSystem.get(conf);
    testdata = getTestTempDirPath("testdata");
    output = getTestTempDirPath("output");
    // Create small reference data set
    referenceData = TestKmeansClustering.getPointsWritable(REFERENCE);
    // generate larger test data set for the clustering tests to chew on
    generateSamples();
  }
  
  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sd
   *          double standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sd) {
    log.info("Generating {} samples m=[{}, {}] sd={}", num, mx, my, sd);
    for (int i = 0; i < num; i++) {
      sampleData.add(new VectorWritable(new DenseVector(new double[] {UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd)})));
    }
  }
  
  private void generateSamples() {
    generateSamples(500, 1, 1, 3);
    generateSamples(300, 1, 0, 0.5);
    generateSamples(300, 0, 2, 0.1);
  }
  
  private void printRepPoints(int numIterations) {
    RepresentativePointsDriver.printRepresentativePoints(output, numIterations);
  }
  
  /**
   * Initialize synthetic data using 4 clusters dC units from origin having 4 representative points dP from each center
   * 
   * @param dC
   *          a double cluster center offset
   * @param dP
   *          a double representative point offset
   * @param measure
   *          the DistanceMeasure
   */
  private void initData(double dC, double dP, DistanceMeasure measure) {
    clusters = Lists.newArrayList();
    clusters.add(new Canopy(new DenseVector(new double[] {-dC, -dC}), 1, measure));
    clusters.add(new Canopy(new DenseVector(new double[] {-dC, dC}), 3, measure));
    clusters.add(new Canopy(new DenseVector(new double[] {dC, dC}), 5, measure));
    clusters.add(new Canopy(new DenseVector(new double[] {dC, -dC}), 7, measure));
    representativePoints = Maps.newHashMap();
    for (Cluster cluster : clusters) {
      List<VectorWritable> points = Lists.newArrayList();
      representativePoints.put(cluster.getId(), points);
      points.add(new VectorWritable(cluster.getCenter().clone()));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] {dP, dP}))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] {dP, -dP}))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] {-dP, -dP}))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] {-dP, dP}))));
    }
  }
  
  @Test
  public void testRepresentativePoints() throws Exception {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Configuration conf = getConfiguration();
    // run using MR reference point calculation
    CanopyDriver.run(conf, testdata, output, measure, 3.1, 1.1, true, 0.0, true);
    int numIterations = 2;
    Path clustersIn = new Path(output, "clusters-0-final");
    RepresentativePointsDriver.run(conf, clustersIn, new Path(output, "clusteredPoints"), output, measure,
        numIterations, false);
    printRepPoints(numIterations);
    ClusterEvaluator evaluatorMR = new ClusterEvaluator(conf, clustersIn);
    // now run again using sequential reference point calculation
    HadoopUtil.delete(conf, output);
    CanopyDriver.run(conf, testdata, output, measure, 3.1, 1.1, true, 0.0, true);
    RepresentativePointsDriver.run(conf, clustersIn, new Path(output, "clusteredPoints"), output, measure,
        numIterations, true);
    printRepPoints(numIterations);
    ClusterEvaluator evaluatorSeq = new ClusterEvaluator(conf, clustersIn);
    // compare results
    assertEquals("InterCluster Density", evaluatorMR.interClusterDensity(), evaluatorSeq.interClusterDensity(), EPSILON);
    assertEquals("IntraCluster Density", evaluatorMR.intraClusterDensity(), evaluatorSeq.intraClusterDensity(), EPSILON);
  }
  
  @Test
  public void testCluster0() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.25, measure);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.33333333333333315, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  @Test
  public void testCluster1() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.5, measure);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.33333333333333315, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  @Test
  public void testCluster2() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.75, measure);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.33333333333333315, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  /**
   * adding an empty cluster should modify the inter cluster density but not change the intra-cluster density as that
   * cluster would have NaN as its intra-cluster density and NaN values are ignored by the evaluator
   * 
   * @throws IOException
   */
  @Test
  public void testEmptyCluster() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.25, measure);
    Canopy cluster = new Canopy(new DenseVector(new double[] {10, 10}), 19, measure);
    clusters.add(cluster);
    List<VectorWritable> points = Lists.newArrayList();
    representativePoints.put(cluster.getId(), points);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.371534146934532, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  /**
   * adding an single-valued cluster should modify the inter cluster density but not change the intra-cluster density as
   * that cluster would have NaN as its intra-cluster density and NaN values are ignored by the evaluator
   * 
   * @throws IOException
   */
  @Test
  public void testSingleValueCluster() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.25, measure);
    Canopy cluster = new Canopy(new DenseVector(new double[] {0, 0}), 19, measure);
    clusters.add(cluster);
    List<VectorWritable> points = Lists.newArrayList();
    points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] {1, 1}))));
    representativePoints.put(cluster.getId(), points);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.3656854249492381, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  /**
   * Representative points extraction will duplicate the cluster center if the cluster has no assigned points. These
   * clusters are included in the inter-cluster density but their NaN intra-density values are ignored by the evaluator.
   * 
   * @throws IOException
   */
  @Test
  public void testAllSameValueCluster() throws IOException {
    ClusteringTestUtils.writePointsToFile(referenceData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.25, measure);
    Canopy cluster = new Canopy(new DenseVector(new double[] {0, 0}), 19, measure);
    clusters.add(cluster);
    List<VectorWritable> points = Lists.newArrayList();
    points.add(new VectorWritable(cluster.getCenter()));
    points.add(new VectorWritable(cluster.getCenter()));
    points.add(new VectorWritable(cluster.getCenter()));
    representativePoints.put(cluster.getId(), points);
    ClusterEvaluator evaluator = new ClusterEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.3656854249492381, evaluator.interClusterDensity(), EPSILON);
    assertEquals("intra cluster density", 0.3656854249492381, evaluator.intraClusterDensity(), EPSILON);
  }
  
  @Test
  public void testCanopy() throws Exception {
    ClusteringTestUtils.writePointsToFile(sampleData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Configuration conf = getConfiguration();
    CanopyDriver.run(conf, testdata, output, measure, 3.1, 1.1, true, 0.0, true);
    int numIterations = 10;
    Path clustersIn = new Path(output, "clusters-0-final");
    RepresentativePointsDriver.run(conf, clustersIn, new Path(output, "clusteredPoints"), output, measure,
        numIterations, true);
    //printRepPoints(numIterations);
    ClusterEvaluator evaluator = new ClusterEvaluator(conf, clustersIn);
    // now print out the Results
    System.out.println("Intra-cluster density = " + evaluator.intraClusterDensity());
    System.out.println("Inter-cluster density = " + evaluator.interClusterDensity());
  }
  
  @Test
  public void testKmeans() throws Exception {
    ClusteringTestUtils.writePointsToFile(sampleData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    // now run the Canopy job to prime kMeans canopies
    Configuration conf = getConfiguration();
    CanopyDriver.run(conf, testdata, output, measure, 3.1, 1.1, false, 0.0, true);
    // now run the KMeans job
    Path kmeansOutput = new Path(output, "kmeans");
    KMeansDriver.run(testdata, new Path(output, "clusters-0-final"), kmeansOutput, 0.001, 10, true, 0.0, true);
    int numIterations = 10;
    Path clustersIn = new Path(kmeansOutput, "clusters-2");
    RepresentativePointsDriver.run(conf, clustersIn, new Path(kmeansOutput, "clusteredPoints"), kmeansOutput, measure,
        numIterations, true);
    RepresentativePointsDriver.printRepresentativePoints(kmeansOutput, numIterations);
    ClusterEvaluator evaluator = new ClusterEvaluator(conf, clustersIn);
    // now print out the Results
    System.out.println("Intra-cluster density = " + evaluator.intraClusterDensity());
    System.out.println("Inter-cluster density = " + evaluator.interClusterDensity());
  }
  
  @Test
  public void testFuzzyKmeans() throws Exception {
    ClusteringTestUtils.writePointsToFile(sampleData, new Path(testdata, "file1"), fs, conf);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    // now run the Canopy job to prime kMeans canopies
    Configuration conf = getConfiguration();
    CanopyDriver.run(conf, testdata, output, measure, 3.1, 1.1, false, 0.0, true);
    Path fuzzyKMeansOutput = new Path(output, "fuzzyk");
    // now run the KMeans job
    FuzzyKMeansDriver.run(testdata, new Path(output, "clusters-0-final"), fuzzyKMeansOutput, 0.001, 10, 2,
        true, true, 0, true);
    int numIterations = 10;
    Path clustersIn = new Path(fuzzyKMeansOutput, "clusters-4");
    RepresentativePointsDriver.run(conf, clustersIn, new Path(fuzzyKMeansOutput, "clusteredPoints"), fuzzyKMeansOutput,
        measure, numIterations, true);
    RepresentativePointsDriver.printRepresentativePoints(fuzzyKMeansOutput, numIterations);
    ClusterEvaluator evaluator = new ClusterEvaluator(conf, clustersIn);
    // now print out the Results
    System.out.println("Intra-cluster density = " + evaluator.intraClusterDensity());
    System.out.println("Inter-cluster density = " + evaluator.interClusterDensity());
  }
  
}
