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

import junit.framework.TestCase;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.DummyOutputCollector;
import org.apache.mahout.utils.EuclideanDistanceMeasure;
import org.apache.mahout.utils.ManhattanDistanceMeasure;
import org.apache.mahout.utils.UserDefinedDistanceMeasure;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.nio.charset.Charset;

public class TestCanopyCreation extends TestCase {
  static final double[][] raw = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 },
      { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 5, 5 } };

  List<Canopy> referenceManhattan;

  final DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  List<Vector> manhattanCentroids;

  List<Canopy> referenceEuclidean;

  final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  List<Vector> euclideanCentroids;

  public TestCanopyCreation(String name) {
    super(name);
  }

  private static List<Vector> getPoints(double[][] raw) {
    List<Vector> points = new ArrayList<Vector>();
    for (double[] fr : raw) {
      Vector vec = new SparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  private static List<Text> getFormattedPoints(List<Vector> points) {
    List<Text> result = new ArrayList<Text>();
    for (Vector point : points) {
      result.add(new Text(point.asFormatString()));
    }
    return result;
  }

  /**
   * Verify that the given canopies are equivalent to the referenceManhattan
   * 
   * @param canopies
   */
  private void verifyManhattanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceManhattan);
  }

  /**
   * Verify that the given canopies are equivalent to the referenceEuclidean
   * 
   * @param canopies
   */
  private void verifyEuclideanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceEuclidean);
  }

  /**
   * Verify that the given canopies are equivalent to the reference. This means
   * the number of canopies is the same, the number of points in each is the
   * same and the centroids are the same.
   * 
   * @param canopies
   */
  private static void verifyCanopies(List<Canopy> canopies, List<Canopy> reference) {
    assertEquals("number of canopies", reference.size(), canopies.size());
    for (int canopyIx = 0; canopyIx < canopies.size(); canopyIx++) {
      Canopy refCanopy = reference.get(canopyIx);
      Canopy testCanopy = canopies.get(canopyIx);
      assertEquals("canopy points " + canopyIx, refCanopy.getNumPoints(),
          testCanopy.getNumPoints());
      Vector refCentroid = refCanopy.computeCentroid();
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.cardinality(); pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid.get(pointIx), testCentroid.get(pointIx));
      }
    }
  }

  /**
   * Print the canopies to the transcript
   * 
   * @param canopies
   *            a List<Canopy>
   */
  private static void printCanopies(List<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      System.out.println(canopy.toString());
    }
  }

  private static void writePointsToFile(List<Vector> points, String fileName)
      throws IOException {
    writePointsToFileWithPayload(points, fileName, "");
  }

  private static void writePointsToFileWithPayload(List<Vector> points,
      String fileName, String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), Charset.forName("UTF-8")));
    for (Vector point : points) {
      output.write(point.asFormatString());
      output.write(payload);
      output.write('\n');
    }
    output.flush();
    output.close();
  }

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
    referenceManhattan = populateCanopies(manhattanDistanceMeasure,
        getPoints(raw), 3.1, 2.1);
    manhattanCentroids = populateCentroids(referenceManhattan);
    referenceEuclidean = populateCanopies(euclideanDistanceMeasure,
        getPoints(raw), 3.1, 2.1);
    euclideanCentroids = populateCentroids(referenceEuclidean);
  }

  /**
   * Iterate through the canopies, adding their centroids to a list
   * 
   * @param canopies
   *            a List<Canopy>
   * @return the List<Vector>
   */
  static List<Vector> populateCentroids(List<Canopy> canopies) {
    List<Vector> result = new ArrayList<Vector>();
    for (Canopy canopy : canopies)
      result.add(canopy.computeCentroid());
    return result;
  }

  /**
   * Iterate through the points, adding new canopies. Return the canopies.
   * 
   * @param measure
   *            a DistanceMeasure to use
   * @param points
   *            a list<Vector> defining the points to be clustered
   * @param t1
   *            the T1 distance threshold
   * @param t2
   *            the T2 distance threshold
   * @return the List<Canopy> created
   */
  static List<Canopy> populateCanopies(DistanceMeasure measure, List<Vector> points,
      double t1, double t2) {
    List<Canopy> canopies = new ArrayList<Canopy>();
    Canopy.config(measure, t1, t2);
    /**
     * Reference Implementation: Given a distance metric, one can create
     * canopies as follows: Start with a list of the data points in any order,
     * and with two distance thresholds, T1 and T2, where T1 > T2. (These
     * thresholds can be set by the user, or selected by cross-validation.) Pick
     * a point on the list and measure its distance to all other points. Put all
     * points that are within distance threshold T1 into a canopy. Remove from
     * the list all points that are within distance threshold T2. Repeat until
     * the list is empty.
     */
    while (!points.isEmpty()) {
      Iterator<Vector> ptIter = points.iterator();
      Vector p1 = ptIter.next();
      ptIter.remove();
      Canopy canopy = new VisibleCanopy(p1);
      canopies.add(canopy);
      while (ptIter.hasNext()) {
        Vector p2 = ptIter.next();
        double dist = measure.distance(p1, p2);
        // Put all points that are within distance threshold T1 into the canopy
        if (dist < t1)
          canopy.addPoint(p2);
        // Remove from the list all points that are within distance threshold T2
        if (dist < t2)
          ptIter.remove();
      }
    }
    return canopies;
  }


  /**
   * Story: User can cluster points using a ManhattanDistanceMeasure and a
   * reference implementation
   * 
   * @throws Exception
   */
  public void testReferenceManhattan() throws Exception {
    System.out.println("testReferenceManhattan");
    // see setUp for cluster creation
    printCanopies(referenceManhattan);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceManhattan.get(canopyIx);
      int[] expectedNumPoints = { 4, 4, 3 };
      double[][] expectedCentroids = { { 1.5, 1.5 }, { 4.0, 4.0 },
          { 4.666666666666667, 4.6666666666666667 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /**
   * Story: User can cluster points using a EuclideanDistanceMeasure and a
   * reference implementation
   * 
   * @throws Exception
   */
  public void testReferenceEuclidean() throws Exception {
    System.out.println("testReferenceEuclidean()");
    // see setUp for cluster creation
    printCanopies(referenceEuclidean);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceEuclidean.get(canopyIx);
      int[] expectedNumPoints = { 5, 5, 3 };
      double[][] expectedCentroids = { { 1.8, 1.8 }, { 4.2, 4.2 },
          { 4.666666666666667, 4.666666666666667 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /**
   * Story: User can cluster points without instantiating them all in memory at
   * once
   * 
   * @throws Exception
   */
  public void testIterativeManhattan() throws Exception {
    List<Vector> points = getPoints(raw);
    Canopy.config(new ManhattanDistanceMeasure(), 3.1, 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (Vector point : points)
      Canopy.addPointToCanopies(point, canopies);

    System.out.println("testIterativeManhattan");
    printCanopies(canopies);
    verifyManhattanCanopies(canopies);
  }

  /**
   * Story: User can cluster points without instantiating them all in memory at
   * once
   * 
   * @throws Exception
   */
  public void testIterativeEuclidean() throws Exception {
    List<Vector> points = getPoints(raw);
    Canopy.config(new EuclideanDistanceMeasure(), 3.1, 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (Vector point : points)
      Canopy.addPointToCanopies(point, canopies);

    System.out.println("testIterativeEuclidean");
    printCanopies(canopies);
    verifyEuclideanCanopies(canopies);
  }

  /**
   * Story: User can produce initial canopy centers using a
   * ManhattanDistanceMeasure and a CanopyMapper/Combiner which clusters input
   * points to produce an output set of canopy centroid points.
   * 
   * @throws Exception
   */
  public void testCanopyMapperManhattan() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    CanopyCombiner combiner = new CanopyCombiner();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    Canopy.config(manhattanDistanceMeasure, (3.1), (2.1));
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    assertEquals("Number of map results", 3, collector.getData().size());
    // now combine the mapper output
    Canopy.config(manhattanDistanceMeasure, (3.1), (2.1));
    Map<String, List<Text>> mapData = collector.getData();
    collector = new DummyOutputCollector<Text,Text>();
    for (Map.Entry<String, List<Text>> stringListEntry : mapData.entrySet())
      combiner.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector,
          null);
    // now verify the output
    List<Text> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++)
      assertEquals("Centroid error",
          manhattanCentroids.get(i).asFormatString(), AbstractVector.decodeVector(
              data.get(i).toString()).asFormatString());
  }

  /**
   * Story: User can produce initial canopy centers using a
   * EuclideanDistanceMeasure and a CanopyMapper/Combiner which clusters input
   * points to produce an output set of canopy centroid points.
   * 
   * @throws Exception
   */
  public void testCanopyMapperEuclidean() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    CanopyCombiner combiner = new CanopyCombiner();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    Canopy.config(euclideanDistanceMeasure, (3.1), (2.1));
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    assertEquals("Number of map results", 3, collector.getData().size());
    // now combine the mapper output
    Canopy.config(euclideanDistanceMeasure, (3.1), (2.1));
    Map<String, List<Text>> mapData = collector.getData();
    collector = new DummyOutputCollector<Text,Text>();
    for (Map.Entry<String, List<Text>> stringListEntry : mapData.entrySet())
      combiner.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector,
          null);
    // now verify the output
    List<Text> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++)
      assertEquals("Centroid error",
          euclideanCentroids.get(i).asFormatString(), AbstractVector.decodeVector(
              data.get(i).toString()).asFormatString());
  }

  /**
   * Story: User can produce final canopy centers using a
   * ManhattanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   * 
   * @throws Exception
   */
  public void testCanopyReducerManhattan() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    Canopy.config(manhattanDistanceMeasure, (3.1), (2.1));
    List<Vector> points = getPoints(raw);
    List<Text> texts = getFormattedPoints(points);
    reducer.reduce(new Text("centroid"), texts.iterator(), collector, null);
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Text> data = collector.getValue(key);
      assertEquals("Centroid error",
          manhattanCentroids.get(i).asFormatString(), Canopy.decodeCanopy(
              data.get(0).toString()).getCenter().asFormatString());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a
   * EuclideanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   * 
   * @throws Exception
   */
  public void testCanopyReducerEuclidean() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    Canopy.config(euclideanDistanceMeasure, (3.1), (2.1));
    List<Vector> points = getPoints(raw);
    List<Text> texts = getFormattedPoints(points);
    reducer.reduce(new Text("centroid"), texts.iterator(), collector, null);
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Text> data = collector.getValue(key);
      assertEquals("Centroid error",
          euclideanCentroids.get(i).asFormatString(), Canopy.decodeCanopy(
              data.get(0).toString()).getCenter().asFormatString());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job
   * and a ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testCanopyGenManhattanMR() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies",
        ManhattanDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    JobConf job = new JobConf(
        org.apache.mahout.clustering.canopy.CanopyDriver.class);
    FileSystem fs = FileSystem.get(job);
    Path path = new Path("output/canopies/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Text value = new Text();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C0", key.toString());
    assertEquals("1st value", "C0: [s2, 0:1.5, 1:1.5, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "C1: [s2, 0:4.333333333333334, 1:4.333333333333334, ] ",
        value.toString());
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job
   * and a EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testCanopyGenEuclideanMR() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies",
        EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    JobConf job = new JobConf(
        org.apache.mahout.clustering.canopy.CanopyDriver.class);
    FileSystem fs = FileSystem.get(job);
    Path path = new Path("output/canopies/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Text value = new Text();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C0", key.toString());
    assertEquals("1st value", "C0: [s2, 0:1.8, 1:1.8, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "C1: [s2, 0:4.433333333333334, 1:4.433333333333334, ] ",
        value.toString());
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterMapper and a
   * ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterMapperManhattan() throws Exception {
    Canopy.config(manhattanDistanceMeasure, (3.1), (2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    for (Vector centroid : manhattanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    Map<String, List<Text>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet()) {
      Canopy canopy = Canopy.decodeCanopy(stringListEntry.getKey());
      List<Text> pts = stringListEntry.getValue();
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(AbstractVector.decodeVector(ptDef
            .toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterMapper and a
   * EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterMapperEuclidean() throws Exception {
    Canopy.config(euclideanDistanceMeasure, (3.1), (2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    for (Vector centroid : euclideanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    Map<String, List<Text>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet()) {
      Canopy canopy = Canopy.decodeCanopy(stringListEntry.getKey());
      List<Text> pts = stringListEntry.getValue();
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(AbstractVector.decodeVector(ptDef
            .toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterReducer and a
   * ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterReducerManhattan() throws Exception {
    Canopy.config(manhattanDistanceMeasure, (3.1), (2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    for (Vector centroid : manhattanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    Map<String, List<Text>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());

    // reduce the data
    Reducer<Text, Text, Text, Text> reducer = new IdentityReducer<Text, Text>();
    collector = new DummyOutputCollector<Text,Text>();
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet())
      reducer.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector, null);

    // check the output
    data = collector.getData();
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet()) {
      Canopy canopy = Canopy.decodeCanopy(stringListEntry.getKey());
      List<Text> pts = stringListEntry.getValue();
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(AbstractVector.decodeVector(ptDef
            .toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterReducer and a
   * EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterReducerEuclidean() throws Exception {
    Canopy.config(euclideanDistanceMeasure, (3.1), (2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text,Text> collector = new DummyOutputCollector<Text,Text>();
    for (Vector centroid : euclideanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Vector> points = getPoints(raw);
    // map the data
    for (Vector point : points)
      mapper.map(new Text(), new Text(point.asFormatString()), collector, null);
    Map<String, List<Text>> data = collector.getData();

    // reduce the data
    Reducer<Text, Text, Text, Text> reducer = new IdentityReducer<Text, Text>();
    collector = new DummyOutputCollector<Text,Text>();
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet())
      reducer.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector, null);

    // check the output
    data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<Text>> stringListEntry : data.entrySet()) {
      Canopy canopy = Canopy.decodeCanopy(stringListEntry.getKey());
      List<Text> pts = stringListEntry.getValue();
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(AbstractVector.decodeVector(ptDef
            .toString())));
    }
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusteringManhattanMR() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        ManhattanDistanceMeasure.class.getName(), 3.1, 2.1);
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(
        "output/clusters/part-00000"), Charset.forName("UTF-8")));
    int count = 0;
    while (reader.ready()) {
      System.out.println(reader.readLine());
      count++;
    }
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusteringEuclideanMR() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(
        "output/clusters/part-00000"), Charset.forName("UTF-8")));
    int count = 0;
    while (reader.ready()) {
      System.out.println(reader.readLine());
      count++;
    }
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a ManhattanDistanceMeasure. Input points can have extra payload
   * information following the point [...] and this information will be retained
   * in the output.
   * 
   * @throws Exception
   */
  public void testClusteringManhattanMRWithPayload() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFileWithPayload(points, "testdata/file1", "file1");
    writePointsToFileWithPayload(points, "testdata/file2", "file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        ManhattanDistanceMeasure.class.getName(), 3.1, 2.1);
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(
        "output/clusters/part-00000"), Charset.forName("UTF-8")));
    int count = 0;
    while (reader.ready()) {
      String line = reader.readLine();
      assertTrue("No payload", line.indexOf("file") > 0);
      System.out.println(line);
      count++;
    }
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure. Input points can have extra payload
   * information following the point [...] and this information will be retained
   * in the output.
   * 
   * @throws Exception
   */
  public void testClusteringEuclideanMRWithPayload() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFileWithPayload(points, "testdata/file1", "file1");
    writePointsToFileWithPayload(points, "testdata/file2", "file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(
        "output/clusters/part-00000"), Charset.forName("UTF-8")));
    int count = 0;
    while (reader.ready()) {
      String line = reader.readLine();
      assertTrue("No payload", line.indexOf("file") > 0);
      System.out.println(line);
      count++;
    }
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }

  /**
   * Story: Clustering algorithm must support arbitrary user defined distance
   * measure
   * 
   * @throws Exception
   */
  public void testUserDefinedDistanceMeasure() throws Exception {
    List<Vector> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver. User defined measure happens to be a Manhattan
    // subclass so results are same.
    CanopyDriver.runJob("testdata", "output/canopies",
        UserDefinedDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    JobConf job = new JobConf(
        org.apache.mahout.clustering.canopy.CanopyDriver.class);
    FileSystem fs = FileSystem.get(job);
    Path path = new Path("output/canopies/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Text value = new Text();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C0", key.toString());
    assertEquals("1st value", "C0: [s2, 0:1.5, 1:1.5, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "C1: [s2, 0:4.333333333333334, 1:4.333333333333334, ] ",
        value.toString());
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }
}
