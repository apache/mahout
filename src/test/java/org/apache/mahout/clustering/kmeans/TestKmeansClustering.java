/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.canopy.DummyOutputCollector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.EuclideanDistanceMeasure;
import org.apache.mahout.utils.ManhattanDistanceMeasure;
import org.apache.mahout.utils.Point;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class TestKmeansClustering extends TestCase {

  static final float[][] reference = {{1, 1}, {2, 1}, {1, 2}, {2, 2},
          {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};

  static int[][] expectedNumPoints = {{9}, {4, 5}, {4, 5, 0},
          {1, 2, 1, 5}, {1, 1, 1, 2, 4}, {1, 1, 1, 1, 1, 4},
          {1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 2, 1},
          {1, 1, 1, 1, 1, 1, 1, 1, 1}};

  private void rmr(String path) throws Exception {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (int i = 0; i < contents.length; i++)
          rmr(f.toString() + File.separator + contents[i]);
      }
      f.delete();
    }
  }

  protected void setUp() throws Exception {
    super.setUp();
    rmr("output");
    rmr("testdata");
  }

  @Override
  protected void tearDown() throws Exception {
    super.tearDown();
  }

  /**
   * This is the reference k-means implementation. Given its inputs it iterates
   * over the points and clusters until their centers converge or until the
   * maximum number of iterations is exceeded.
   *
   * @param points   the input List<Float[]> of points
   * @param clusters the initial List<Cluster> of clusters
   * @param measure  the DistanceMeasure to use
   * @param maxIter  the maximum number of iterations
   */
  private void referenceKmeans(List<Float[]> points, List<Cluster> clusters,
                               DistanceMeasure measure, int maxIter) {
    boolean converged = false;
    int iteration = 0;
    while (!converged && iteration++ < maxIter) {
      converged = iterateReference(points, clusters, measure);
    }
  }

  /**
   * Perform a single iteration over the points and clusters, assigning points
   * to clusters and returning if the iterations are completed.
   *
   * @param points   the List<Float[]> having the input points
   * @param clusters the List<Cluster> clusters
   * @param measure  a DistanceMeasure to use
   * @return
   */
  private boolean iterateReference(List<Float[]> points,
                                   List<Cluster> clusters, DistanceMeasure measure) {
    boolean converged;
    converged = true;
    // iterate through all points, assigning each to the nearest cluster
    for (Float[] point : points) {
      Cluster closestCluster = null;
      float closestDistance = Float.MAX_VALUE;
      for (Cluster cluster : clusters) {
        float distance = measure.distance(cluster.getCenter(), point);
        if (closestCluster == null || closestDistance > distance) {
          closestCluster = cluster;
          closestDistance = distance;
        }
      }
      closestCluster.addPoint(point);
    }
    // test for convergence
    for (Cluster cluster : clusters) {
      if (!cluster.computeConvergence())
        converged = false;
    }
    // update the cluster centers
    if (!converged)
      for (Cluster cluster : clusters)
        cluster.recomputeCenter();
    return converged;
  }

  private List<Float[]> getPoints(float[][] raw) {
    List<Float[]> points = new ArrayList<Float[]>();
    for (int i = 0; i < raw.length; i++) {
      float[] fr = raw[i];
      Float[] fs = new Float[fr.length];
      for (int j = 0; j < fs.length; j++)
        fs[j] = fr[j];
      points.add(fs);
    }
    return points;
  }

  /**
   * Story: Test the reference implementation
   *
   * @throws Exception
   */
  public void testReferenceImplementation() throws Exception {
    List<Float[]> points = getPoints(reference);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Cluster.config(measure, (float) 0.001);
    // try all possible values of k
    for (int k = 0; k < points.size(); k++) {
      System.out.println("Test k=" + (k + 1) + ":");
      // pick k initial cluster centers at random
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++)
        clusters.add(new VisibleCluster(points.get(i)));
      // iterate clusters until they converge
      int maxIter = 10;
      referenceKmeans(points, clusters, measure, maxIter);
      for (int c = 0; c < clusters.size(); c++) {
        Cluster cluster = clusters.get(c);
        assertEquals("Cluster " + c + " test " + k, expectedNumPoints[k][c],
                cluster.getNumPoints());
        System.out.println(cluster.toString());
      }
    }
  }

  /**
   * Story: test that the mapper will map input points to the nearest cluster
   *
   * @throws Exception
   */
  public void testKMeansMapper() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, (float) 0.001);
    List<Float[]> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyOutputCollector collector = new DummyOutputCollector();
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i));
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.config(clusters);
      // map the data
      for (Float[] point : points)
        mapper.map(new Text(), new Text(Point.formatPoint(point)), collector,
                null);
      assertEquals("Number of map results", k + 1, collector.getData().size());
      // now verify that all points are correctly allocated
      for (String key : collector.getKeys()) {
        Cluster cluster = Cluster.decodeCluster(key);
        List<Text> values = collector.getValue(key);
        for (Writable value : values) {
          Float[] point = Point.decodePoint(value.toString());
          float distance = euclideanDistanceMeasure.distance(cluster
                  .getCenter(), point);
          for (Cluster c : clusters)
            assertTrue("distance error", distance <= euclideanDistanceMeasure
                    .distance(point, c.getCenter()));
        }
      }
    }
  }

  /**
   * Story: test that the combiner will produce partial cluster totals for all
   * of the clusters and points that it sees
   *
   * @throws Exception
   */
  public void testKMeansCombiner() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, (float) 0.001);
    List<Float[]> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyOutputCollector collector = new DummyOutputCollector();
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i));
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.config(clusters);
      // map the data
      for (Float[] point : points)
        mapper.map(new Text(), new Text(Point.formatPoint(point)), collector,
                null);

      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyOutputCollector collector2 = new DummyOutputCollector();
      for (String key : collector.getKeys())
        combiner.reduce(new Text(key), collector.getValue(key).iterator(),
                collector2, null);

      assertEquals("Number of map results", k + 1, collector2.getData().size());
      // now verify that all points are accounted for
      int count = 0;
      Float[] total = Point.origin(2);
      for (String key : collector2.getKeys()) {
        List<Text> values = collector2.getValue(key);
        assertEquals("too many values", 1, values.size());
        String value = values.get(0).toString();
        int ix = value.indexOf(",");
        count += new Integer(value.substring(0, ix));
        total = Point.sum(total, Point.decodePoint(value.substring(ix + 2)));
      }
      assertEquals("total points", 9, count);
      assertEquals("point total[0]", 27, total[0].intValue());
      assertEquals("point total[1]", 27, total[1].intValue());
    }
  }

  /**
   * Story: test that the reducer will sum the partial cluster totals for all of
   * the clusters and points that it sees
   *
   * @throws Exception
   */
  public void testKMeansReducer() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
    Cluster.config(euclideanDistanceMeasure, (float) 0.001);
    List<Float[]> points = getPoints(reference);
    for (int k = 0; k < points.size(); k++) {
      System.out.println("K = " + k);
      // pick k initial cluster centers at random
      DummyOutputCollector collector = new DummyOutputCollector();
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i), i);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.config(clusters);
      // map the data
      for (Float[] point : points)
        mapper.map(new Text(), new Text(Point.formatPoint(point)), collector,
                null);

      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyOutputCollector collector2 = new DummyOutputCollector();
      for (String key : collector.getKeys())
        combiner.reduce(new Text(key), collector.getValue(key).iterator(),
                collector2, null);

      // now reduce the data
      KMeansReducer reducer = new KMeansReducer();
      DummyOutputCollector collector3 = new DummyOutputCollector();
      for (String key : collector2.getKeys())
        reducer.reduce(new Text(key), collector2.getValue(key).iterator(),
                collector3, null);

      assertEquals("Number of map results", k + 1, collector3.getData().size());

      // compute the reference result after one iteration and compare
      List<Cluster> reference = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++)
        reference.add(new Cluster(points.get(i), i));
      boolean converged = iterateReference(points, reference,
              euclideanDistanceMeasure);
      if (k == 8)
        assertTrue("not converged? " + k, converged);
      else
        assertFalse("converged? " + k, converged);

      // now verify that all clusters have correct centers
      converged = true;
      for (int i = 0; i < reference.size(); i++) {
        Cluster ref = reference.get(i);
        String key = ref.getIdentifier();
        List<Text> values = collector3.getValue(key);
        String value = values.get(0).toString();
        Cluster cluster = Cluster.decodeCluster(value);
        converged = converged && cluster.isConverged();
        System.out.println("ref= " + ref.toString() + " cluster= "
                + cluster.toString());
        assertEquals(k + " center[" + key + "][0]", ref.getCenter()[0], cluster
                .getCenter()[0]);
        assertEquals(k + " center[" + key + "][1]", ref.getCenter()[1], cluster
                .getCenter()[1]);
      }
      if (k == 8)
        assertTrue("not converged? " + k, converged);
      else
        assertFalse("converged? " + k, converged);
    }
  }

  /**
   * Story: User wishes to run kmeans job on reference data
   *
   * @throws Exception
   */
  public void testKMeansMRJob() throws Exception {
    List<Float[]> points = getPoints(reference);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    testData = new File("testdata/points");
    if (!testData.exists())
      testData.mkdir();
    Point.writePointsToFile(points, "testdata/points/file1");
    Point.writePointsToFile(points, "testdata/points/file2");
    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      JobConf job = new JobConf(KMeansDriver.class);
      FileSystem fs = FileSystem.get(job);
      Path path = new Path("testdata/clusters/part-00000");
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path,
              Text.class, Text.class);
      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i));
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        writer.append(new Text(cluster.getIdentifier()), new Text(Cluster
                .formatCluster(cluster)));
      }
      writer.close();

      // now run the Job
      KMeansDriver.runJob("testdata/points", "testdata/clusters", "output",
              EuclideanDistanceMeasure.class.getName(), "0.001", "10");

      // now compare the expected clusters with actual
      File outDir = new File("output/points");
      assertTrue("output dir exists?", outDir.exists());
      String[] outFiles = outDir.list();
      assertEquals("output dir files?", 4, outFiles.length);
      BufferedReader reader = new BufferedReader(new FileReader(
              "output/points/part-00000"));
      int[] expect = expectedNumPoints[k];
      DummyOutputCollector collector = new DummyOutputCollector();
      while (reader.ready()) {
        String line = reader.readLine();
        String[] lineParts = line.split("\t");
        assertEquals("line parts", 2, lineParts.length);
        String cl = line.substring(0, line.indexOf(':'));
        collector.collect(new Text(cl), new Text(lineParts[1]));
      }
      reader.close();
      if (k == 2)
        // cluster 3 is empty so won't appear in output
        assertEquals("clusters[" + k + "]", expect.length - 1, collector
                .getKeys().size());
      else
        assertEquals("clusters[" + k + "]", expect.length, collector.getKeys()
                .size());
    }
  }

  /**
   * Story: User wants to use canopy clustering to input the initial clusters
   * for kmeans job.
   *
   * @throws Exception
   */
  public void textKMeansWithCanopyClusterInput() throws Exception {
    List<Float[]> points = getPoints(reference);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    testData = new File("testdata/points");
    if (!testData.exists())
      testData.mkdir();
    Point.writePointsToFile(points, "testdata/points/file1");
    Point.writePointsToFile(points, "testdata/points/file2");

    // now run the Canopy job
    CanopyDriver.runJob("testdata/points", "testdata/canopies",
            ManhattanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1);

    // now run the KMeans job
    KMeansDriver.runJob("testdata/points", "testdata/canopies", "output",
            EuclideanDistanceMeasure.class.getName(), "0.001", "10");

    // now compare the expected clusters with actual
    File outDir = new File("output/points");
    assertTrue("output dir exists?", outDir.exists());
    String[] outFiles = outDir.list();
    assertEquals("output dir files?", 4, outFiles.length);
    BufferedReader reader = new BufferedReader(new FileReader(
            "output/points/part-00000"));
    DummyOutputCollector collector = new DummyOutputCollector();
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
