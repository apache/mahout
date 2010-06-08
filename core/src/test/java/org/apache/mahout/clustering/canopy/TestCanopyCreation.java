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

package org.apache.mahout.clustering.canopy;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.MockMapperContext;
import org.apache.mahout.clustering.MockReducerContext;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.UserDefinedDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestCanopyCreation extends MahoutTestCase {

  private static final double[][] raw = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 5, 5 } };

  private List<Canopy> referenceManhattan;

  private final DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  private List<Vector> manhattanCentroids;

  private List<Canopy> referenceEuclidean;

  private final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  private List<Vector> euclideanCentroids;

  private FileSystem fs;

  private static List<VectorWritable> getPointsWritable() {
    List<VectorWritable> points = new ArrayList<VectorWritable>();
    for (double[] fr : raw) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  private static List<Vector> getPoints() {
    List<Vector> points = new ArrayList<Vector>();
    for (double[] fr : raw) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  /** Verify that the given canopies are equivalent to the referenceManhattan */
  private void verifyManhattanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceManhattan);
  }

  /** Verify that the given canopies are equivalent to the referenceEuclidean */
  private void verifyEuclideanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceEuclidean);
  }

  /**
   * Verify that the given canopies are equivalent to the reference. This means the number of canopies is the
   * same, the number of points in each is the same and the centroids are the same.
   */
  private static void verifyCanopies(List<Canopy> canopies, List<Canopy> reference) {
    assertEquals("number of canopies", reference.size(), canopies.size());
    for (int canopyIx = 0; canopyIx < canopies.size(); canopyIx++) {
      Canopy refCanopy = reference.get(canopyIx);
      Canopy testCanopy = canopies.get(canopyIx);
      assertEquals("canopy points " + canopyIx, refCanopy.getNumPoints(), testCanopy.getNumPoints());
      Vector refCentroid = refCanopy.computeCentroid();
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.size(); pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']', refCentroid.get(pointIx), testCentroid.get(pointIx));
      }
    }
  }

  /**
   * Print the canopies to the transcript
   * 
   * @param canopies
   *          a List<Canopy>
   */
  private static void printCanopies(List<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      System.out.println(canopy.toString());
    }
  }

  public static void rmr(String path) {
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
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
    rmr("output");
    rmr("testdata");
    referenceManhattan = CanopyClusterer.createCanopies(getPoints(), manhattanDistanceMeasure, 3.1, 2.1);
    manhattanCentroids = CanopyClusterer.calculateCentroids(referenceManhattan);
    referenceEuclidean = CanopyClusterer.createCanopies(getPoints(), euclideanDistanceMeasure, 3.1, 2.1);
    euclideanCentroids = CanopyClusterer.calculateCentroids(referenceEuclidean);
  }

  /** Story: User can cluster points using a ManhattanDistanceMeasure and a reference implementation */
  public void testReferenceManhattan() throws Exception {
    System.out.println("testReferenceManhattan");
    // see setUp for cluster creation
    printCanopies(referenceManhattan);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceManhattan.get(canopyIx);
      int[] expectedNumPoints = { 4, 4, 3 };
      double[][] expectedCentroids = { { 1.5, 1.5 }, { 4.0, 4.0 }, { 4.666666666666667, 4.6666666666666667 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx], testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']', refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /** Story: User can cluster points using a EuclideanDistanceMeasure and a reference implementation */
  public void testReferenceEuclidean() throws Exception {
    System.out.println("testReferenceEuclidean()");
    // see setUp for cluster creation
    printCanopies(referenceEuclidean);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceEuclidean.get(canopyIx);
      int[] expectedNumPoints = { 5, 5, 3 };
      double[][] expectedCentroids = { { 1.8, 1.8 }, { 4.2, 4.2 }, { 4.666666666666667, 4.666666666666667 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx], testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']', refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /** Story: User can cluster points without instantiating them all in memory at once */
  public void testIterativeManhattan() throws Exception {
    List<Vector> points = getPoints();
    List<Canopy> canopies = CanopyClusterer.createCanopies(points, new ManhattanDistanceMeasure(), 3.1, 2.1);
    System.out.println("testIterativeManhattan");
    printCanopies(canopies);
    verifyManhattanCanopies(canopies);
  }

  /** Story: User can cluster points without instantiating them all in memory at once */
  public void testIterativeEuclidean() throws Exception {
    List<Vector> points = getPoints();
    List<Canopy> canopies = CanopyClusterer.createCanopies(points, new EuclideanDistanceMeasure(), 3.1, 2.1);

    System.out.println("testIterativeEuclidean");
    printCanopies(canopies);
    verifyEuclideanCanopies(canopies);
  }

  /**
   * Story: User can produce initial canopy centers using a ManhattanDistanceMeasure and a
   * CanopyMapper/Combiner which clusters input points to produce an output set of canopy centroid points.
   */
  public void testCanopyMapperManhattan() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    MockMapperContext<Text, VectorWritable> context = new MockMapperContext<Text, VectorWritable>(mapper, conf, collector);
    mapper.setup(context);

    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    mapper.cleanup(context);
    assertEquals("Number of map results", 1, collector.getData().size());
    // now verify the output
    List<VectorWritable> data = collector.getValue(new Text("centroid"));
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error", manhattanCentroids.get(i).asFormatString(), data.get(i).get().asFormatString());
    }
  }

  /**
   * Story: User can produce initial canopy centers using a EuclideanDistanceMeasure and a
   * CanopyMapper/Combiner which clusters input points to produce an output set of canopy centroid points.
   */
  public void testCanopyMapperEuclidean() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    MockMapperContext<Text, VectorWritable> context = new MockMapperContext<Text, VectorWritable>(mapper, conf, collector);
    mapper.setup(context);

    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    mapper.cleanup(context);
    assertEquals("Number of map results", 1, collector.getData().size());
    // now verify the output
    List<VectorWritable> data = collector.getValue(new Text("centroid"));
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error", euclideanCentroids.get(i).asFormatString(), data.get(i).get().asFormatString());
    }
  }

  /**
   * Story: User can produce final canopy centers using a ManhattanDistanceMeasure and a CanopyReducer which
   * clusters input centroid points to produce an output set of final canopy centroid points.
   */
  public void testCanopyReducerManhattan() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<Text, Canopy> collector = new DummyOutputCollector<Text, Canopy>();
    MockReducerContext<Text, Canopy> context = new MockReducerContext<Text, Canopy>(reducer, conf, collector, Text.class,
        Canopy.class);
    reducer.setup(context);

    List<VectorWritable> points = getPointsWritable();
    reducer.reduce(new Text("centroid"), points, context);
    Set<Text> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (Text key : keys) {
      List<Canopy> data = collector.getValue(key);
      assertEquals(manhattanCentroids.get(i).asFormatString() + " is not equal to "
          + data.get(0).computeCentroid().asFormatString(), manhattanCentroids.get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a EuclideanDistanceMeasure and a CanopyReducer which
   * clusters input centroid points to produce an output set of final canopy centroid points.
   */
  public void testCanopyReducerEuclidean() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<Text, Canopy> collector = new DummyOutputCollector<Text, Canopy>();
    MockReducerContext<Text, Canopy> context = new MockReducerContext<Text, Canopy>(reducer, conf, collector, Text.class,
        Canopy.class);
    reducer.setup(context);

    List<VectorWritable> points = getPointsWritable();
    reducer.reduce(new Text("centroid"), points, context);
    Set<Text> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (Text key : keys) {
      List<Canopy> data = collector.getValue(key);
      assertEquals(euclideanCentroids.get(i).asFormatString() + " is not equal to "
          + data.get(0).computeCentroid().asFormatString(), euclideanCentroids.get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job and a
   * ManhattanDistanceMeasure.
   */
  public void testCanopyGenManhattanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration config = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file1"), fs, config);
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file2"), fs, config);
    // now run the Canopy Driver
    Path output = getTestTempDirPath("output");
    CanopyDriver.runJob(getTestTempDirPath("testdata"), output, ManhattanDistanceMeasure.class.getName(), 3.1, 2.1, false);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), config);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, config);
    Text key = new Text();
    Canopy canopy = new Canopy();
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("1st key", "C-0", key.toString());
    // Canopy canopy = new Canopy(value); //Canopy.decodeCanopy(value.toString());
    assertEquals("1st x value", 1.5, canopy.getCenter().get(0));
    assertEquals("1st y value", 1.5, canopy.getCenter().get(1));
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("2nd key", "C-1", key.toString());
    // canopy = Canopy.decodeCanopy(canopy.toString());
    assertEquals("1st x value", 4.333333333333334, canopy.getCenter().get(0));
    assertEquals("1st y value", 4.333333333333334, canopy.getCenter().get(1));
    assertFalse("more to come", reader.next(key, canopy));
    reader.close();
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job and a
   * EuclideanDistanceMeasure.
   */
  public void testCanopyGenEuclideanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration job = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file1"), fs, job);
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file2"), fs, job);
    // now run the Canopy Driver
    Path output = getTestTempDirPath("output");
    CanopyDriver.runJob(getTestTempDirPath("testdata"), output, EuclideanDistanceMeasure.class.getName(), 3.1, 2.1, false);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C-0", key.toString());
    assertEquals("1st x value", 1.8, value.getCenter().get(0));
    assertEquals("1st y value", 1.8, value.getCenter().get(1));
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C-1", key.toString());
    assertEquals("1st x value", 4.433333333333334, value.getCenter().get(0));
    assertEquals("1st y value", 4.433333333333334, value.getCenter().get(1));
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /** Story: User can cluster a subset of the points using a ClusterMapper and a ManhattanDistanceMeasure. */
  public void testClusterMapperManhattan() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<IntWritable, WeightedVectorWritable> collector = new DummyOutputCollector<IntWritable, WeightedVectorWritable>();
    MockMapperContext<IntWritable, WeightedVectorWritable> context = new MockMapperContext<IntWritable, WeightedVectorWritable>(
        mapper, conf, collector);
    mapper.setup(context);

    List<Canopy> canopies = new ArrayList<Canopy>();
    int nextCanopyId = 0;
    for (Vector centroid : manhattanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    Map<IntWritable, List<WeightedVectorWritable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Entry<IntWritable, List<WeightedVectorWritable>> stringListEntry : data.entrySet()) {
      IntWritable key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key.get(), canopies);
      List<WeightedVectorWritable> pts = stringListEntry.getValue();
      for (WeightedVectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.getVector().get()));
      }
    }
  }

  private static Canopy findCanopy(Integer key, List<Canopy> canopies) {
    for (Canopy c : canopies) {
      if (c.getId() == key) {
        return c;
      }
    }
    return null;
  }

  /** Story: User can cluster a subset of the points using a ClusterMapper and a EuclideanDistanceMeasure. */
  public void testClusterMapperEuclidean() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyOutputCollector<IntWritable, WeightedVectorWritable> collector = new DummyOutputCollector<IntWritable, WeightedVectorWritable>();
    MockMapperContext<IntWritable, WeightedVectorWritable> context = new MockMapperContext<IntWritable, WeightedVectorWritable>(
        mapper, conf, collector);
    mapper.setup(context);

    List<Canopy> canopies = new ArrayList<Canopy>();
    int nextCanopyId = 0;
    for (Vector centroid : euclideanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    Map<IntWritable, List<WeightedVectorWritable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Entry<IntWritable, List<WeightedVectorWritable>> stringListEntry : data.entrySet()) {
      IntWritable key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key.get(), canopies);
      List<WeightedVectorWritable> pts = stringListEntry.getValue();
      for (WeightedVectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.getVector().get()));
      }
    }
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce job and a
   * ManhattanDistanceMeasure.
   */
  public void testClusteringManhattanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job
    Path output = getTestTempDirPath("output");
    CanopyDriver.runJob(getTestTempDirPath("testdata"), output, ManhattanDistanceMeasure.class.getName(), 3.1, 2.1, true);
    Path path = new Path(output, "clusteredPoints/part-m-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    int count = 0;
    /*
     * while (reader.ready()) { System.out.println(reader.readLine()); count++; }
     */
    IntWritable clusterId = new IntWritable(0);
    WeightedVectorWritable vector = new WeightedVectorWritable();
    while (reader.next(clusterId, vector)) {
      count++;
      System.out.println("Txt: " + clusterId + " Vec: " + vector.getVector().get().asFormatString());
    }
    assertEquals("number of points", points.size(), count);
    reader.close();
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce job and a
   * EuclideanDistanceMeasure.
   */
  public void testClusteringEuclideanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job
    Path output = getTestTempDirPath("output");
    CanopyDriver.runJob(getTestTempDirPath("testdata"), output, EuclideanDistanceMeasure.class.getName(), 3.1, 2.1, true);
    Path path = new Path(output, "clusteredPoints/part-m-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    int count = 0;
    IntWritable canopyId = new IntWritable(0);
    WeightedVectorWritable can = new WeightedVectorWritable();
    while (reader.next(canopyId, can)) {
      count++;
    }
    assertEquals("number of points", points.size(), count);
    reader.close();
  }

  /** Story: Clustering algorithm must support arbitrary user defined distance measure */
  public void testUserDefinedDistanceMeasure() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Canopy Driver. User defined measure happens to be a Manhattan
    // subclass so results are same.
    Path output = getTestTempDirPath("output");
    CanopyDriver.runJob(getTestTempDirPath("testdata"), output, UserDefinedDistanceMeasure.class.getName(), 3.1, 2.1, false);

    // verify output from sequence file
    Configuration job = new Configuration();
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C-0", key.toString());

    assertEquals("1st x value", 1.5, value.getCenter().get(0));
    assertEquals("1st y value", 1.5, value.getCenter().get(1));
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C-1", key.toString());

    assertEquals("1st x value", 4.333333333333334, value.getCenter().get(0));
    assertEquals("1st y value", 4.333333333333334, value.getCenter().get(1));
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }
}
