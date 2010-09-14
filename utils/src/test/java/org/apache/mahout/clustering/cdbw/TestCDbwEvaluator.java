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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyDriver;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.MahoutTestCase;
import org.junit.Before;
import org.junit.Test;

public final class TestCDbwEvaluator extends MahoutTestCase {

  private static final double[][] REFERENCE = {
      { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 5, 5 } };

  private Map<Integer, List<VectorWritable>> representativePoints;
  private Map<Integer, Cluster> clusters;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // Create test data
    List<VectorWritable> sampleData = TestKmeansClustering.getPointsWritable(REFERENCE);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("testdata/file1"), fs, conf);
  }

  private void checkRefPoints(int numIterations) throws IOException {
    for (int i = 0; i <= numIterations; i++) {
      Path out = new Path(getTestTempDirPath("output"), "representativePoints-" + i);
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      for (FileStatus file : fs.listStatus(out)) {
        if (!file.getPath().getName().startsWith(".")) {
          SequenceFile.Reader reader = new SequenceFile.Reader(fs, file.getPath(), conf);
          try {
            Writable clusterId = new IntWritable(0);
            VectorWritable point = new VectorWritable();
            while (reader.next(clusterId, point)) {
              System.out.println("\tC-" + clusterId + ": " + AbstractCluster.formatVector(point.get(), null));
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
   * @param measure TODO
   */
  private void initData(double dC, double dP, DistanceMeasure measure) {
    clusters = new HashMap<Integer, Cluster>();
    clusters.put(1, new Canopy(new DenseVector(new double[] { -dC, -dC }), 1, measure));
    clusters.put(3, new Canopy(new DenseVector(new double[] { -dC, dC }), 3, measure));
    clusters.put(5, new Canopy(new DenseVector(new double[] { dC, dC }), 5, measure));
    clusters.put(7, new Canopy(new DenseVector(new double[] { dC, -dC }), 7, measure));
    representativePoints = new HashMap<Integer, List<VectorWritable>>();
    for (Cluster cluster : clusters.values()) {
      List<VectorWritable> points = new ArrayList<VectorWritable>();
      representativePoints.put(cluster.getId(), points);
      points.add(new VectorWritable(cluster.getCenter().clone()));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { dP, dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { dP, -dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { -dP, -dP }))));
      points.add(new VectorWritable(cluster.getCenter().plus(new DenseVector(new double[] { -dP, dP }))));
    }
  }

  @Test
  public void testCDbw0() {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.25, measure);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.0, evaluator.interClusterDensity(), EPSILON);
    assertEquals("separation", 1.5, evaluator.separation(), EPSILON);
    assertEquals("intra cluster density", 0.8944271909999157, evaluator.intraClusterDensity(), EPSILON);
    assertEquals("CDbw", 1.3416407864998736, evaluator.getCDbw(), EPSILON);
  }

  @Test
  public void testCDbw1() {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.5, measure);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 0.0, evaluator.interClusterDensity(), EPSILON);
    assertEquals("separation", 1.0, evaluator.separation(), EPSILON);
    assertEquals("intra cluster density", 0.44721359549995787, evaluator.intraClusterDensity(), EPSILON);
    assertEquals("CDbw", 0.44721359549995787, evaluator.getCDbw(), EPSILON);
  }

  @Test
  public void testCDbw2() {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    initData(1, 0.75, measure);
    CDbwEvaluator evaluator = new CDbwEvaluator(representativePoints, clusters, measure);
    assertEquals("inter cluster density", 1.017921815355728, evaluator.interClusterDensity(), EPSILON);
    assertEquals("separation", 0.24777966925931558, evaluator.separation(), EPSILON);
    assertEquals("intra cluster density", 0.29814239699997197, evaluator.intraClusterDensity(), EPSILON);
    assertEquals("CDbw", 0.07387362452083261, evaluator.getCDbw(), EPSILON);
  }

  @Test
  public void testCanopy() throws Exception { // now run the Job
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    CanopyDriver.runJob(getTestTempDirPath("testdata"), getTestTempDirPath("output"), measure, 3.1, 2.1, true, false);
    int numIterations = 2;
    Path output = getTestTempDirPath("output");
    CDbwDriver.runJob(new Path(output, "clusters-0"), new Path(output, "clusteredPoints"), output, measure, numIterations, 1);
    checkRefPoints(numIterations);
  }

  @Test
  public void testKmeans() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob(getTestTempDirPath("testdata"), getTestTempDirPath("output"), measure, 3.1, 2.1, false, false);
    // now run the KMeans job
    Path output = getTestTempDirPath("output");
    KMeansDriver.runJob(getTestTempDirPath("testdata"), new Path(output, "clusters-0"), output, measure, 0.001, 10, 1, true, false);
    int numIterations = 2;
    CDbwDriver.runJob(new Path(output, "clusters-2"), new Path(output, "clusteredPoints"), output, measure, numIterations, 1);
    checkRefPoints(numIterations);
  }

  @Test
  public void testFuzzyKmeans() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob(getTestTempDirPath("testdata"), getTestTempDirPath("output"), measure, 3.1, 2.1, false, false);
    // now run the KMeans job
    Path output = getTestTempDirPath("output");
    FuzzyKMeansDriver.runJob(getTestTempDirPath("testdata"),
                             new Path(output, "clusters-0"),
                             output,
                             measure,
                             0.001,
                             10,
                             1,
                             2,
                             true,
                             true,
                             0,
                             false);
    int numIterations = 2;
    CDbwDriver.runJob(new Path(output, "clusters-4"), new Path(output, "clusteredPoints"), output, measure, numIterations, 1);
    checkRefPoints(numIterations);
  }

  @Test
  public void testMeanShift() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    MeanShiftCanopyDriver.runJob(getTestTempDirPath("testdata"),
                                 getTestTempDirPath("output"),
                                 measure,
                                 2.1,
                                 1.0,
                                 0.001,
                                 10,
                                 false,
                                 true,
                                 false);
    int numIterations = 2;
    Path output = getTestTempDirPath("output");
    CDbwDriver.runJob(new Path(output, "clusters-2"), new Path(output, "clusteredPoints"), output, measure, numIterations, 1);
    checkRefPoints(numIterations);
  }

  @Test
  public void testDirichlet() throws Exception {
    ModelDistribution<VectorWritable> modelDistribution =
        new GaussianClusterDistribution(new VectorWritable(new DenseVector(2)));
    DirichletDriver.runJob(getTestTempDirPath("testdata"),
                           getTestTempDirPath("output"),
                           modelDistribution,
                           15,
                           5,
                           1.0,
                           1,
                           true,
                           true,
                           0,
                           true);
    int numIterations = 2;
    Path output = getTestTempDirPath("output");
    CDbwDriver.runJob(new Path(output, "clusters-5"),
                      new Path(output, "clusteredPoints"),
                      output,
                      new EuclideanDistanceMeasure(),
                      numIterations,
                      1);
    checkRefPoints(numIterations);
  }

}
