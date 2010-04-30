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

package org.apache.mahout.clustering.cdbw;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.models.L1ModelDistribution;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyJob;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestCDbwEvaluator extends MahoutTestCase {

  public static final double[][] reference = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 },
      { 5, 5 } };

  private List<VectorWritable> sampleData;

  private Map<Integer, List<VectorWritable>> representativePoints;

  Map<Integer, Cluster> clusters;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // Create testdata directory
    ClusteringTestUtils.rmr("testdata");
    File f = new File("testdata");
    f.mkdir();
    ClusteringTestUtils.rmr("output");
    // Create test data
    sampleData = TestKmeansClustering.getPointsWritable(reference);
    ClusteringTestUtils.writePointsToFile(sampleData, "testdata/file1", fs, conf);
  }

  private void checkRefPoints(int numIterations) throws IOException {
    File out = new File("output");
    assertTrue("output is not Dir", out.isDirectory());
    for (int i = 0; i <= numIterations; i++) {
      out = new File("output/representativePoints-" + i);
      assertTrue("rep-i is not a Dir", out.isDirectory());
      System.out.println(out.getName() + ":");
      File[] files = out.listFiles();
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      for (File file : files) {
        if (!file.getName().startsWith(".")) {
          SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(file.getAbsolutePath()), conf);
          try {
            IntWritable clusterId = new IntWritable(0);
            VectorWritable point = new VectorWritable();
            while (reader.next(clusterId, point)) {
              System.out.println("\tC-" + clusterId + ": " + ClusterBase.formatVector(point.get(), null));
            }
          } finally {
            reader.close();
          }
        }
      }
    }
  }

  /**
   * Initialize synthetic data using 4 clusters dC units from origin having 4 representative points dP from each center
   * @param dC a double cluster center offset
   * @param dP a double representative point offset
   */
  private void initData(double dC, double dP) {
    clusters = new HashMap<Integer, Cluster>();
    clusters.put(1, new Canopy(new DenseVector(new double[] { -dC, -dC }), 1));
    clusters.put(3, new Canopy(new DenseVector(new double[] { -dC, dC }), 3));
    clusters.put(5, new Canopy(new DenseVector(new double[] { dC, dC }), 5));
    clusters.put(7, new Canopy(new DenseVector(new double[] { dC, -dC }), 7));
    representativePoints = new HashMap<Integer, List<VectorWritable>>();
    for (Cluster cluster : clusters.values()) {
      ArrayList<VectorWritable> points = new ArrayList<VectorWritable>();
      representativePoints.put(cluster.getId(), points);
      points.add(new VectorWritable(cluster.getCenter().clone()));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { dP, dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { dP, -dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { -dP, -dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { -dP, dP }))));
    }
  }

  public void testCDbw0() {
    initData(1, 0.25);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, new EuclideanDistanceMeasure());
    assertEquals("inter cluster density", 0.0, evaluator.interClusterDensity());
    assertEquals("separation", 1.5, evaluator.separation());
    assertEquals("intra cluster density", 0.8944271909999157, evaluator.intraClusterDensity());
    assertEquals("CDbw", 1.3416407864998736, evaluator.CDbw());
  }

  public void testCDbw1() {
    initData(1, 0.5);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, new EuclideanDistanceMeasure());
    assertEquals("inter cluster density", 0.0, evaluator.interClusterDensity());
    assertEquals("separation", 1.0, evaluator.separation());
    assertEquals("intra cluster density", 0.44721359549995787, evaluator.intraClusterDensity());
    assertEquals("CDbw", 0.44721359549995787, evaluator.CDbw());
  }

  public void testCDbw2() {
    initData(1, 0.75);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, new EuclideanDistanceMeasure());
    assertEquals("inter cluster density", 1.017921815355728, evaluator.interClusterDensity());
    assertEquals("separation", 0.24777966925931558, evaluator.separation());
    assertEquals("intra cluster density", 0.29814239699997197, evaluator.intraClusterDensity());
    assertEquals("CDbw", 0.07387362452083261, evaluator.CDbw());
  }

  public void testCanopy() throws Exception { // now run the Job
    CanopyDriver.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1, true);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-0", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(),
        numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testKmeans() throws Exception {
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1, false);
    // now run the KMeans job
    KMeansDriver.runJob("testdata", "output/clusters-0", "output", EuclideanDistanceMeasure.class.getName(), 0.001, 10, 1);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-2", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(),
        numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testFuzzyKmeans() throws Exception {
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1, false);
    // now run the KMeans job
    FuzzyKMeansDriver.runJob("testdata", "output/clusters-0", "output", EuclideanDistanceMeasure.class.getName(), 0.001, 10, 1, 1,
        2, false, true, 0);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-4", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(),
        numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testMeanShift() throws Exception {
    MeanShiftCanopyJob.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 2.1, 1.0, 0.001, 10);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-2", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(),
        numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testDirichlet() throws Exception {
    Vector prototype = new DenseVector(2);
    DirichletDriver.runJob("testdata", "output", L1ModelDistribution.class.getName(), prototype.getClass().getName(), prototype
        .size(), 15, 5, 1.0, 1, true, true, 0);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-5", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(),
        numIterations, 1);
    checkRefPoints(numIterations);
  }

}
