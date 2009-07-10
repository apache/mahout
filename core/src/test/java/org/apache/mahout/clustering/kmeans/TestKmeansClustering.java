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


import junit.framework.TestCase;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.DummyOutputCollector;
import org.apache.mahout.utils.EuclideanDistanceMeasure;
import org.apache.mahout.utils.HadoopUtil;
import org.apache.mahout.utils.ManhattanDistanceMeasure;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestKmeansClustering extends TestCase {

  public static final double[][] reference = {{1, 1}, {2, 1}, {1, 2},
      {2, 2}, {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};

  public static final int[][] expectedNumPoints = {{9}, {4, 5},
      {4, 4, 1}, {1, 2, 1, 5}, {1, 1, 1, 2, 4}, {1, 1, 1, 1, 1, 4},
      {1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 2, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1}};

  FileSystem fs;

  private static void rmr(String path) throws Exception {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (String content : contents) {
          rmr(f.toString() + File.separator + content);
        }
      }
      f.delete();
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rmr("output");
    rmr("testdata");
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }

  /**
   * This is the reference k-means implementation. Given its inputs it iterates over the points and clusters until their
   * centers converge or until the maximum number of iterations is exceeded.
   *
   * @param points   the input List<Vector> of points
   * @param clusters the initial List<Cluster> of clusters
   * @param measure  the DistanceMeasure to use
   * @param maxIter  the maximum number of iterations
   */
  private void referenceKmeans(List<Vector> points, List<Cluster> clusters,
                               DistanceMeasure measure, int maxIter) {
    boolean converged = false;
    int iteration = 0;
    while (!converged && iteration++ < maxIter) {
      converged = iterateReference(points, clusters, measure);
    }
  }

  /**
   * Perform a single iteration over the points and clusters, assigning points to clusters and returning if the
   * iterations are completed.
   *
   * @param points   the List<Vector> having the input points
   * @param clusters the List<Cluster> clusters
   * @param measure  a DistanceMeasure to use
   */
  private static boolean iterateReference(List<Vector> points, List<Cluster> clusters,
                                   DistanceMeasure measure) {
    // iterate through all points, assigning each to the nearest cluster
    for (Vector point : points) {
      Cluster closestCluster = null;
      double closestDistance = Double.MAX_VALUE;
      for (Cluster cluster : clusters) {
        double distance = measure.distance(cluster.getCenter(), point);
        if (closestCluster == null || closestDistance > distance) {
          closestCluster = cluster;
          closestDistance = distance;
        }
      }
      closestCluster.addPoint(point);
    }
    // test for convergence
    boolean converged = true;
    for (Cluster cluster : clusters) {
      if (!cluster.computeConvergence()) {
        converged = false;
      }
    }
    // update the cluster centers
    if (!converged) {
      for (Cluster cluster : clusters) {
        cluster.recomputeCenter();
      }
    }
    return converged;
  }

  public static List<Vector> getPoints(double[][] raw) {
    List<Vector> points = new ArrayList<Vector>();
    for (int i = 0; i < raw.length; i++) {
      double[] fr = raw[i];
      Vector vec = new SparseVector(String.valueOf(i), fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  /** Story: Test the reference implementation */
  public void testReferenceImplementation() throws Exception {
    List<Vector> points = getPoints(reference);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Cluster.config(measure, 0.001);
    // try all possible values of k
    for (int k = 0; k < points.size(); k++) {
      System.out.println("Test k=" + (k + 1) + ':');
      // pick k initial cluster centers at random
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);
        clusters.add(new VisibleCluster(vec));
      }
      // iterate clusters until they converge
      int maxIter = 10;
      referenceKmeans(points, clusters, measure, maxIter);
      for (int c = 0; c < clusters.size(); c++) {
        Cluster cluster = clusters.get(c);
        System.out.println(cluster.toString());
        assertEquals("Cluster " + c + " test " + (k + 1), expectedNumPoints[k][c],
            cluster.getNumPoints());
      }
    }
  }

  private static Map<String, Cluster> loadClusterMap(List<Cluster> clusters) {
    Map<String, Cluster> clusterMap = new HashMap<String, Cluster>();

    for (Cluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    return clusterMap;
  }

  /** Story: test that the mapper will map input points to the nearest cluster */
  public void testKMeansMapper() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, 0.001);
    List<Vector> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyOutputCollector<Text, KMeansInfo> collector = new DummyOutputCollector<Text, KMeansInfo>();
      List<Cluster> clusters = new ArrayList<Cluster>();

      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i));
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }

      Map<String, Cluster> clusterMap = loadClusterMap(clusters);
      mapper.config(clusters);
      // map the data
      for (Vector point : points) {
        mapper.map(new Text(), point, collector,
            null);
      }
      assertEquals("Number of map results", k + 1, collector.getData().size());
      // now verify that all points are correctly allocated
      for (String key : collector.getKeys()) {
        Cluster cluster = clusterMap.get(key);
        List<KMeansInfo> values = collector.getValue(key);
        for (KMeansInfo value : values) {
          double distance = euclideanDistanceMeasure.distance(cluster
              .getCenter(), value.getPointTotal());
          for (Cluster c : clusters) {
            assertTrue("distance error", distance <= euclideanDistanceMeasure
                .distance(value.getPointTotal(), c.getCenter()));
          }
        }
      }
    }
  }

  /** Story: test that the combiner will produce partial cluster totals for all of the clusters and points that it sees */
  public void testKMeansCombiner() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, 0.001);
    List<Vector> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyOutputCollector<Text, KMeansInfo> collector = new DummyOutputCollector<Text, KMeansInfo>();
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);

        Cluster cluster = new Cluster(vec);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.config(clusters);
      // map the data
      for (Vector point : points) {
        mapper.map(new Text(), point, collector,
            null);
      }
      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyOutputCollector<Text, KMeansInfo> collector2 = new DummyOutputCollector<Text, KMeansInfo>();
      for (String key : collector.getKeys()) {
        combiner.reduce(new Text(key), collector.getValue(key).iterator(),
            collector2, null);
      }

      assertEquals("Number of map results", k + 1, collector2.getData().size());
      // now verify that all points are accounted for
      int count = 0;
      Vector total = new DenseVector(2);
      for (String key : collector2.getKeys()) {
        List<KMeansInfo> values = collector2.getValue(key);
        assertEquals("too many values", 1, values.size());
        //String value = values.get(0).toString();
        KMeansInfo info = values.get(0);

        count += info.getPoints();
        total = total.plus(info.getPointTotal());
      }
      assertEquals("total points", 9, count);
      assertEquals("point total[0]", 27, (int) total.get(0));
      assertEquals("point total[1]", 27, (int) total.get(1));
    }
  }

  /** Story: test that the reducer will sum the partial cluster totals for all of the clusters and points that it sees */
  public void testKMeansReducer() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, 0.001);
    List<Vector> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      System.out.println("K = " + k);
      // pick k initial cluster centers at random
      DummyOutputCollector<Text, KMeansInfo> collector = new DummyOutputCollector<Text, KMeansInfo>();
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);
        Cluster cluster = new Cluster(vec, i);
        // add the center so the centroid will be correct upon output
        // cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.config(clusters);
      // map the data
      for (Vector point : points) {
        mapper.map(new Text(), point, collector,
            null);
      }
      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyOutputCollector<Text, KMeansInfo> collector2 = new DummyOutputCollector<Text, KMeansInfo>();
      for (String key : collector.getKeys()) {
        combiner.reduce(new Text(key), collector.getValue(key).iterator(),
            collector2, null);
      }

      // now reduce the data
      KMeansReducer reducer = new KMeansReducer();
      reducer.config(clusters);
      DummyOutputCollector<Text, Cluster> collector3 = new DummyOutputCollector<Text, Cluster>();
      for (String key : collector2.getKeys()) {
        reducer.reduce(new Text(key), collector2.getValue(key).iterator(),
            collector3, null);
      }

      assertEquals("Number of map results", k + 1, collector3.getData().size());

      // compute the reference result after one iteration and compare
      List<Cluster> reference = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);
        reference.add(new Cluster(vec, i));
      }
      boolean converged = iterateReference(points, reference,
          euclideanDistanceMeasure);
      if (k == 8) {
        assertTrue("not converged? " + k, converged);
      } else {
        assertFalse("converged? " + k, converged);
      }

      // now verify that all clusters have correct centers
      converged = true;
      for (int i = 0; i < reference.size(); i++) {
        Cluster ref = reference.get(i);
        String key = ref.getIdentifier();
        List<Cluster> values = collector3.getValue(key);
        Cluster cluster = values.get(0);
        converged = converged && cluster.isConverged();
        //Since we aren't roundtripping through Writable, we need to compare the reference center with the cluster centroid
        cluster.recomputeCenter();
        assertEquals(i + " reference center: " + ref.getCenter().asFormatString() + " and cluster center:  "
            + cluster.getCenter().asFormatString() + " are not equal", ref.getCenter(), cluster.getCenter());

        /*assertEquals(k + " center[" + key + "][1]", ref.getCenter().get(1),
            cluster.getCenter().get(1));*/
      }
      if (k == 8) {
        assertTrue("not converged? " + k, converged);
      } else {
        assertFalse("converged? " + k, converged);
      }
    }
  }

  /** Story: User wishes to run kmeans job on reference data */
  public void testKMeansMRJob() throws Exception {
    List<Vector> points = getPoints(reference);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    testData = new File("testdata/points");
    if (!testData.exists()) {
      testData.mkdir();
    }

    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, "testdata/points/file1", fs, conf);
    ClusteringTestUtils.writePointsToFile(points, "testdata/points/file2", fs, conf);
    for (int k = 1; k < points.size(); k++) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      JobConf job = new JobConf(KMeansDriver.class);
      Path path = new Path("testdata/clusters/part-00000");
      FileSystem fs = FileSystem.get(path.toUri(), job);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path,
          Text.class, Cluster.class);

      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);

        Cluster cluster = new Cluster(vec, i);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        writer.append(new Text(cluster.getIdentifier()), cluster);
      }
      writer.close();
      // now run the Job
      HadoopUtil.overwriteOutput("output");
      KMeansDriver.runJob("testdata/points", "testdata/clusters", "output",
          EuclideanDistanceMeasure.class.getName(), 0.001, 10, k + 1, SparseVector.class);
      // now compare the expected clusters with actual
      File outDir = new File("output/points");
      assertTrue("output dir exists?", outDir.exists());
      // assertEquals("output dir files?", 4, outFiles.length);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("output/points/part-00000"), conf);
      int[] expect = expectedNumPoints[k];
      DummyOutputCollector<Text, Text> collector = new DummyOutputCollector<Text, Text>();
      //The key is the name of the vector, or the vector itself
      Text key = new Text();
      //The value is the cluster id
      Text value = new Text();
      while (reader.next(key, value)) {
        /*String line = reader.readLine();
        String[] lineParts = line.split("\t");
        assertEquals("line parts", 2, lineParts.length);*/
        // String cl = line.substring(0, line.indexOf(':'));
        //collector.collect(new Text(lineParts[1]), new Text(lineParts[0]));
        collector.collect(value, key);
        key = new Text();
        value = new Text();

      }
      reader.close();
      if (k == 2)
      // cluster 3 is empty so won't appear in output
      {
        assertEquals("clusters[" + k + ']', expect.length - 1, collector
            .getKeys().size());
      } else {
        assertEquals("clusters[" + k + ']', expect.length, collector.getKeys()
            .size());
      }
    }
  }

  /** Story: User wants to use canopy clustering to input the initial clusters for kmeans job. */
  public void textKMeansWithCanopyClusterInput() throws Exception {
    List<Vector> points = getPoints(reference);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    testData = new File("testdata/points");
    if (!testData.exists()) {
      testData.mkdir();
    }
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, "testdata/points/file1", fs, conf);
    ClusteringTestUtils.writePointsToFile(points, "testdata/points/file2", fs, conf);

    // now run the Canopy job
    CanopyDriver.runJob("testdata/points", "testdata/canopies",
        ManhattanDistanceMeasure.class.getName(), 3.1, 2.1, SparseVector.class);

    // now run the KMeans job
    KMeansDriver.runJob("testdata/points", "testdata/canopies", "output",
        EuclideanDistanceMeasure.class.getName(), 0.001, 10, 1, SparseVector.class);

    // now compare the expected clusters with actual
    File outDir = new File("output/points");
    assertTrue("output dir exists?", outDir.exists());
    String[] outFiles = outDir.list();
    assertEquals("output dir files?", 4, outFiles.length);
    BufferedReader reader = new BufferedReader(new InputStreamReader(
        new FileInputStream("output/points/part-00000"), Charset
            .forName("UTF-8")));
    DummyOutputCollector<Text, Text> collector = new DummyOutputCollector<Text, Text>();
    while (reader.ready()) {
      String line = reader.readLine();
      String[] lineParts = line.split("\t");
      assertEquals("line parts", 2, lineParts.length);
      String cl = line.substring(0, line.indexOf(':'));
      collector.collect(new Text(cl), new Text(lineParts[1]));
    }
    reader.close();
    assertEquals("num points[V0]", 4, collector.getValue("V0").size());
    assertEquals("num points[V1]", 5, collector.getValue("V1").size());
  }


}
