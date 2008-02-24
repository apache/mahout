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
package org.apache.mahout.clustering.canopy;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import junit.framework.TestCase;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.EuclideanDistanceMeasure;
import org.apache.mahout.utils.ManhattanDistanceMeasure;
import org.apache.mahout.utils.UserDefinedDistanceMeasure;

public class TestCanopyCreation extends TestCase {
  static final float[][] raw = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 },
      { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 5, 5 } };

  List<Canopy> referenceManhattan;

  DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  List<Float[]> manhattanCentroids;

  List<Canopy> referenceEuclidean;

  DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  List<Float[]> euclideanCentroids;

  public TestCanopyCreation(String name) {
    super(name);
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

  private List<Text> getFormattedPoints(List<Float[]> points) {
    List<Text> result = new ArrayList<Text>();
    for (Float[] point : points)
      result.add(new Text(Canopy.formatPoint(point)));
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
  private void verifyCanopies(List<Canopy> canopies, List<Canopy> reference) {
    assertEquals("number of canopies", reference.size(), canopies.size());
    for (int canopyIx = 0; canopyIx < canopies.size(); canopyIx++) {
      Canopy refCanopy = reference.get(canopyIx);
      Canopy testCanopy = canopies.get(canopyIx);
      assertEquals("canopy points " + canopyIx, refCanopy.getNumPoints(),
          testCanopy.getNumPoints());
      Float[] refCentroid = refCanopy.computeCentroid();
      Float[] testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + "[" + pointIx + "]",
            refCentroid[pointIx], testCentroid[pointIx]);
      }
    }
  }

  /**
   * Print the canopies to the transcript
   * 
   * @param canopies a List<Canopy>
   */
  private void prtCanopies(List<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      System.out.println(canopy.toString());
    }
  }

  private void writePointsToFile(List<Float[]> points, String fileName)
      throws IOException {
    writePointsToFileWithPayload(points, fileName, "");
  }

  private void writePointsToFileWithPayload(List<Float[]> points,
      String fileName, String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new FileWriter(fileName));
    for (Float[] point : points) {
      output.write(Canopy.formatPoint(point));
      output.write(payload);
      output.write("\n");
    }
    output.flush();
    output.close();
  }

  protected void setUp() throws Exception {
    super.setUp();
    referenceManhattan = populateCanopies(manhattanDistanceMeasure,
        getPoints(raw), (float) 3.1, (float) 2.1);
    manhattanCentroids = populateCentroids(referenceManhattan);
    referenceEuclidean = populateCanopies(euclideanDistanceMeasure,
        getPoints(raw), (float) 3.1, (float) 2.1);
    euclideanCentroids = populateCentroids(referenceEuclidean);
  }

  /**
   * Iterate through the canopies, adding their centroids to a list
   * 
   * @param canopies a List<Canopy>
   * @return the List<Float[]>
   */
  List<Float[]> populateCentroids(List<Canopy> canopies) {
    List<Float[]> result = new ArrayList<Float[]>();
    for (Canopy canopy : canopies)
      result.add(canopy.computeCentroid());
    return result;
  }

  /**
   * Iterate through the points, adding new canopies. Return the canopies.
   * 
   * @param measure a DistanceMeasure to use
   * @param points a list<Float[]> defining the points to be clustered
   * @param t1 the T1 distance threshold
   * @param t2 the T2 distance threshold
   * @return the List<Canopy> created
   */
  List<Canopy> populateCanopies(DistanceMeasure measure, List<Float[]> points,
      float t1, float t2) {
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
      Iterator<Float[]> ptIter = points.iterator();
      Float[] p1 = ptIter.next();
      ptIter.remove();
      Canopy canopy = new VisibleCanopy(p1);
      canopies.add(canopy);
      while (ptIter.hasNext()) {
        Float[] p2 = ptIter.next();
        float dist = measure.distance(p1, p2);
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

  protected void tearDown() throws Exception {
    super.tearDown();
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
    prtCanopies(referenceManhattan);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceManhattan.get(canopyIx);
      int[] expectedNumPoints = { 4, 4, 3 };
      float[][] expectedCentroids = { { (float) 1.5, (float) 1.5 },
          { (float) 4.0, (float) 4.0 },
          { (float) 4.6666665, (float) 4.6666665 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      float[] refCentroid = expectedCentroids[canopyIx];
      Float[] testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + "[" + pointIx + "]",
            refCentroid[pointIx], testCentroid[pointIx]);
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
    prtCanopies(referenceEuclidean);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceEuclidean.get(canopyIx);
      int[] expectedNumPoints = { 5, 5, 3 };
      float[][] expectedCentroids = { { (float) 1.8, (float) 1.8 },
          { (float) 4.2, (float) 4.2 },
          { (float) 4.6666665, (float) 4.6666665 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      float[] refCentroid = expectedCentroids[canopyIx];
      Float[] testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + "[" + pointIx + "]",
            refCentroid[pointIx], testCentroid[pointIx]);
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
    List<Float[]> points = getPoints(raw);
    Canopy.config(new ManhattanDistanceMeasure(), (float) 3.1, (float) 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (Float[] point : points)
      Canopy.addPointToCanopies(point, canopies);

    System.out.println("testIterativeManhattan");
    prtCanopies(canopies);
    verifyManhattanCanopies(canopies);
  }

  /**
   * Story: User can cluster points without instantiating them all in memory at
   * once
   * 
   * @throws Exception
   */
  public void testIterativeEuclidean() throws Exception {
    List<Float[]> points = getPoints(raw);
    Canopy.config(new EuclideanDistanceMeasure(), (float) 3.1, (float) 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (Float[] point : points)
      Canopy.addPointToCanopies(point, canopies);

    System.out.println("testIterativeEuclidean");
    prtCanopies(canopies);
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
    DummyOutputCollector collector = new DummyOutputCollector();
    Canopy.config(manhattanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    assertEquals("Number of map results", 3, collector.getData().size());
    // now combine the mapper output
    Canopy.config(manhattanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    Map<String, List<Writable>> mapData = collector.getData();
    collector = new DummyOutputCollector();
    for (String key : mapData.keySet())
      combiner.reduce(new Text(key), mapData.get(key).iterator(), collector,
          null);
    // now verify the output
    List<Writable> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++)
      assertEquals("Centroid error", Canopy.formatPoint(manhattanCentroids
          .get(i)), Canopy.formatPoint(Canopy.decodePoint(data.get(i)
          .toString())));
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
    DummyOutputCollector collector = new DummyOutputCollector();
    Canopy.config(euclideanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    assertEquals("Number of map results", 3, collector.getData().size());
    // now combine the mapper output
    Canopy.config(euclideanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    Map<String, List<Writable>> mapData = collector.getData();
    collector = new DummyOutputCollector();
    for (String key : mapData.keySet())
      combiner.reduce(new Text(key), mapData.get(key).iterator(), collector,
          null);
    // now verify the output
    List<Writable> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++)
      assertEquals("Centroid error", Canopy.formatPoint(euclideanCentroids
          .get(i)), Canopy.formatPoint(Canopy.decodePoint(data.get(i)
          .toString())));
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
    DummyOutputCollector collector = new DummyOutputCollector();
    Canopy.config(manhattanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    List<Float[]> points = getPoints(raw);
    List<Text> texts = getFormattedPoints(points);
    reducer.reduce(new Text("centroid"), texts.iterator(), collector, null);
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Writable> data = collector.getValue(key);
      assertEquals("Centroid error", Canopy.formatPoint(manhattanCentroids
          .get(i)), Canopy.formatPoint(Canopy.decodePoint(data.get(0)
          .toString())));
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
    DummyOutputCollector collector = new DummyOutputCollector();
    Canopy.config(euclideanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    List<Float[]> points = getPoints(raw);
    List<Text> texts = getFormattedPoints(points);
    reducer.reduce(new Text("centroid"), texts.iterator(), collector, null);
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Writable> data = collector.getValue(key);
      assertEquals("Centroid error", Canopy.formatPoint(euclideanCentroids
          .get(i)), Canopy.formatPoint(Canopy.decodePoint(data.get(0)
          .toString())));
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies",
        ManhattanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");

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
    assertEquals("1st value", "[1.5, 1.5, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "[4.333333, 4.333333, ] ", value.toString());
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies",
        EuclideanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");

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
    assertEquals("1st value", "[1.8, 1.8, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "[4.4333334, 4.4333334, ] ", value.toString());
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
    Canopy.config(manhattanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector collector = new DummyOutputCollector();
    for (Float[] centroid : manhattanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    Map<String, List<Writable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (String canopyDef : data.keySet()) {
      Canopy canopy = Canopy.decodeCanopy(canopyDef);
      List<Writable> pts = data.get(canopyDef);
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(Canopy
            .decodePoint(ptDef.toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterMapper and a
   * EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterMapperEuclidean() throws Exception {
    Canopy.config(euclideanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector collector = new DummyOutputCollector();
    for (Float[] centroid : euclideanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    Map<String, List<Writable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (String canopyDef : data.keySet()) {
      Canopy canopy = Canopy.decodeCanopy(canopyDef);
      List<Writable> pts = data.get(canopyDef);
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(Canopy
            .decodePoint(ptDef.toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterReducer and a
   * ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterReducerManhattan() throws Exception {
    Canopy.config(manhattanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector collector = new DummyOutputCollector();
    for (Float[] centroid : manhattanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    Map<String, List<Writable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());

    // reduce the data
    Reducer reducer = new IdentityReducer();
    collector = new DummyOutputCollector();
    for (String key : data.keySet())
      reducer.reduce(new Text(key), data.get(key).iterator(), collector, null);

    // check the output
    data = collector.getData();
    for (String canopyDef : data.keySet()) {
      Canopy canopy = Canopy.decodeCanopy(canopyDef);
      List<Writable> pts = data.get(canopyDef);
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(Canopy
            .decodePoint(ptDef.toString())));
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterReducer and a
   * EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusterReducerEuclidean() throws Exception {
    Canopy.config(euclideanDistanceMeasure, ((float) 3.1), ((float) 2.1));
    ClusterMapper mapper = new ClusterMapper();
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector collector = new DummyOutputCollector();
    for (Float[] centroid : euclideanCentroids)
      canopies.add(new Canopy(centroid));
    mapper.config(canopies);
    List<Float[]> points = getPoints(raw);
    // map the data
    for (Float[] point : points)
      mapper.map(new Text(), new Text(Canopy.formatPoint(point)), collector,
          null);
    Map<String, List<Writable>> data = collector.getData();

    // reduce the data
    Reducer reducer = new IdentityReducer();
    collector = new DummyOutputCollector();
    for (String key : data.keySet())
      reducer.reduce(new Text(key), data.get(key).iterator(), collector, null);

    // check the output
    data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (String canopyDef : data.keySet()) {
      Canopy canopy = Canopy.decodeCanopy(canopyDef);
      List<Writable> pts = data.get(canopyDef);
      for (Writable ptDef : pts)
        assertTrue("Point not in canopy", canopy.covers(Canopy
            .decodePoint(ptDef.toString())));
    }
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a ManhattanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testClusteringManhattanMR() throws Exception {
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        ManhattanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");
    BufferedReader reader = new BufferedReader(new FileReader(
        "output/clusters/part-00000"));
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");
    BufferedReader reader = new BufferedReader(new FileReader(
        "output/clusters/part-00000"));
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFileWithPayload(points, "testdata/file1", "file1");
    writePointsToFileWithPayload(points, "testdata/file2", "file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        ManhattanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");
    BufferedReader reader = new BufferedReader(new FileReader(
        "output/clusters/part-00000"));
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFileWithPayload(points, "testdata/file1", "file1");
    writePointsToFileWithPayload(points, "testdata/file2", "file2");
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");
    BufferedReader reader = new BufferedReader(new FileReader(
        "output/clusters/part-00000"));
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
    List<Float[]> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFile(points, "testdata/file1");
    writePointsToFile(points, "testdata/file2");
    // now run the Canopy Driver. User defined measure happens to be a Manhattan
    // subclass so results are same.
    CanopyDriver.runJob("testdata", "output/canopies",
        UserDefinedDistanceMeasure.class.getName(), (float) 3.1, (float) 2.1, "dist/apache-mahout-0.1-dev.jar");

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
    assertEquals("1st value", "[1.5, 1.5, ] ", value.toString());
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("2nd value", "[4.333333, 4.333333, ] ", value.toString());
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }
}
